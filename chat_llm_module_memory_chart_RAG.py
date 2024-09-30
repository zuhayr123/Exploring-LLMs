import re
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# New imports for embeddings and vector store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ChatLLM:
    def __init__(self, model_name="llama3.1", db_uri="postgresql://abouzuhayr:@localhost:5432/postgres"):
        # Initialize LLM and database
        self.llm = Ollama(model=model_name)
        self.db = SQLDatabase.from_uri(db_uri)
        self.figures = []

        # Initialize memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer")

        # Store the database description
        self.database_description = (
            "The database consists of two tables: `public.employees_table` and `public.departments_table`. This is a PostgreSQL database, so you need to use postgres-related queries.\n\n"
            "The `public.employees_table` table records details about the employees in a company. It includes the following columns:\n"
            "- `EmployeeID`: A unique identifier for each employee.\n"
            "- `FirstName`: The first name of the employee.\n"
            "- `LastName`: The last name of the employee.\n"
            "- `DepartmentID`: A foreign key that links the employee to a department in the `public.departments_table` table.\n"
            "- `Salary`: The salary of the employee.\n\n"
            "The `public.departments_table` table contains information about the various departments in the company. It includes:\n"
            "- `DepartmentID`: A unique identifier for each department.\n"
            "- `DepartmentName`: The name of the department.\n"
            "- `Location`: The location of the department.\n\n"
            "The `DepartmentID` column in the `public.employees_table` table establishes a relationship between the employees and their respective departments in the `public.departments_table` table. This foreign key relationship allows us to join these tables to retrieve detailed information about employees and their departments."
        )

        self.sql_prompt = PromptTemplate(
            input_variables=["database_description", "chat_history", "question"],
            template="""
        {database_description}

        {chat_history}
        Given the above database schema and conversation history, create a syntactically correct SQL query to answer the following question.

        - Include all relevant columns in the SELECT statement.
        - Use double quotes around table and column names to preserve case sensitivity.
        - **Do not include any backslashes or escape characters in the SQL query.**
        - **Provide the SQL query as a plain text without any additional formatting or quotes.**
        - Ensure that the SQL query is compatible with PostgreSQL.
        - Only use the tables and columns listed in the database schema.

        Question: {question}

        Provide the SQL query in the following format:

        SQLQuery:
        SELECT "Column1", "Column2" FROM "public"."Table" WHERE "Condition";

        Now, generate the SQL query to answer the question.
        """
        )


        # Prompt template for answering the question
        self.answer_prompt = PromptTemplate.from_template(
            """Database Description:
        {database_description}

        {chat_history}
        Given the following user question, corresponding SQL query, and SQL result, answer the user question.

        Question: {question}
        SQL Query: {query}
        SQL Result: {result}

        If the SQL Result is "Graph has been generated and stored.", respond only and only with "Here's is the graph you asked for." only.

        Otherwise, provide a detailed answer.

        Answer:"""
        )
        
        # Prompt template for generating plotting code
        self.plot_code_prompt = PromptTemplate(
            input_variables=["sql_result", "question"],
            template="""
        Given the following SQL query result and the user's question, generate Python code using matplotlib and pandas to plot the graph that answers the question. Use the data in the SQL result to create the plot. The SQL result is provided as a variable 'sql_result', which is a list of dictionaries. You can convert it into a pandas DataFrame for plotting. Ensure the code is syntactically correct and uses proper labels and titles.

        SQL Result:
        {sql_result}

        Question:
        {question}

        Provide only the code, without any explanations or additional text.

        Python code:
        import matplotlib.pyplot as plt
        import pandas as pd

        # Assuming sql_result is a list of dictionaries
        df = pd.DataFrame(sql_result)

        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 6))

        # Your plotting code here using 'ax'

        # Do not call plt.show()
        """
        )

        # Create the SQL query chain
        self.write_query = LLMChain(
            llm=self.llm,
            prompt=self.sql_prompt
        )

        # Chain to generate plotting code
        self.generate_plot_code_chain = LLMChain(
            llm=self.llm,
            prompt=self.plot_code_prompt
        )

        self.embeddings = HuggingFaceEmbeddings()
        self.vectorstore = self._initialize_vectorstore("text_doc.txt")

        # Create the LLM chain
        self.chain = self._create_chain()

    # New method to initialize the vector store
    def _initialize_vectorstore(self, document_path):
        # Load and split the text document
        loader = TextLoader(document_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Generate embeddings and create the vector store
        texts_content = [t.page_content for t in texts]
        embeddings = self.embeddings.embed_documents(texts_content)
        vectorstore = FAISS.from_texts(texts_content, self.embeddings)

        return vectorstore
    
    def detect_graph_intent(self, question):
        # Check if the user intends to draw a graph by looking for specific keywords
        graph_keywords = ["graph", "plot", "chart", "visualize"]
        return any(keyword in question.lower() for keyword in graph_keywords)

    def _create_chain(self):
        # New function to decide whether to use the document or database
        def decide_source(inputs):
            question = inputs['question']

            # Check if the question is about personal information
            docs = self.vectorstore.similarity_search(question, k=2)
            if docs:
                print("Relevant documents found.")
                # If relevant documents are found, use them to generate an answer
                context = "\n\n".join([doc.page_content for doc in docs])

                # Create a prompt with the context
                prompt = f"""
                You are an assistant who provides answers based on the following context:

                {context}

                Question: {question}

                Answer:"""
                # Get the answer from the LLM
                answer = self.llm(prompt)
                print(answer)

                # Return the answer directly
                return {'answer': answer, 'question': question}
            else:
                print("No relevant documents found.")
                # Proceed with the existing chain
                return inputs  # Pass inputs to the next step

        decide_source_runnable = RunnableLambda(decide_source)
        
        # Function to generate SQL query with context
        def write_query_with_question(inputs):
            chat_history = self.memory.load_memory_variables({}).get('chat_history', '')
            inputs['chat_history'] = chat_history
            inputs['database_description'] = self.database_description
            response = self.write_query.run(inputs)
            return {'response': response, 'question': inputs['question']}

        write_query_runnable = RunnableLambda(write_query_with_question)

        # Function to extract and execute the SQL query
        def extract_and_execute_sql(inputs):
            response = inputs.get('response', '')
            question = inputs.get('question', '')

            # Print the LLM's response for debugging
            print("LLM Response:")
            print(response)

            # Updated regex pattern
            pattern = re.compile(r'SQLQuery:\s*\n(.*)', re.DOTALL)
            match = pattern.search(response)

            if match:
                sql_query = match.group(1).strip()
                print("Extracted SQL Query:")
                print(sql_query)
                if not sql_query.lower().startswith("select"):
                    result = "Invalid SQL query generated by the LLM."
                else:
                    try:
                        new_result = self.db._execute(sql_query)
                        print(new_result)
                        result = str(new_result)
                    except Exception as e:
                        result = f"Error executing SQL query: {e}"
                # If graph intent is detected, plot the result
                if self.detect_graph_intent(question):
                    # Prepare inputs for plotting code generation
                    plot_code_inputs = {
                        'sql_result': str(new_result),
                        'question': question
                    }
                    # Generate the plotting code
                    plot_code_response = self.generate_plot_code_chain.run(plot_code_inputs)

                    print("Plot Code Response:" + plot_code_response)
                    # Extract the code from the response
                    plot_code = extract_code_from_response(plot_code_response)
                    # Execute the plotting code
                    execute_plot_code(plot_code, new_result)
                    # Return the result indicating that the graph was displayed
                    return {
                        "question": question,
                        "query": sql_query,
                        "result": "Graph has been generated and stored"
                    }
                else:
                    return {
                        "question": question,
                        "query": sql_query,
                        "result": result
                    }
            else:
                print("No SQL query found in the response.")
                return {
                    "question": question,
                    "query": None,
                    "result": "No SQL query found in the response."
                }

        extract_and_execute = RunnableLambda(extract_and_execute_sql)

        def extract_code_from_response(response):
            # Use regex to extract code within code blocks
            code_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
            match = code_pattern.search(response)
            if match:
                code = match.group(1).strip()
            else:
                # If no code block, just return the response
                code = response.strip()
            return code
        
        def execute_plot_code(code, sql_result):
            # Create a local namespace for exec()
            local_vars = {'sql_result': sql_result}
            try:
                exec(code, {}, local_vars)
                fig = local_vars.get('fig', None)
                if fig is None:
                    # Attempt to retrieve 'fig' if not directly assigned
                    fig = local_vars.get('plt', None)
                    if fig and hasattr(fig, 'gcf'):
                        fig = fig.gcf()
                if fig:
                    self.figures.append(fig)
                else:
                    print("No figure object 'fig' was created in the plot code.")
            except Exception as e:
                print(f"Error executing plot code: {e}")

        # Function to add context before generating the final answer
        def add_context(inputs):
            chat_history = self.memory.load_memory_variables({}).get('chat_history', '')
            inputs['chat_history'] = chat_history
            inputs['database_description'] = self.database_description
            return inputs

        add_context_runnable = RunnableLambda(add_context)

        # Combine everything into a chain
        chain = (
            decide_source_runnable|
            write_query_runnable
            | extract_and_execute
            | add_context_runnable
            | self.answer_prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def get_response(self, question):
        # Prepare the inputs
        inputs = {
            "question": question,
        }

        # Call the chain
        response = self.chain.invoke(inputs)

        # Update memory
        self.memory.save_context({"question": question}, {"answer": response})

        return response
