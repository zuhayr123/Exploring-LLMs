import re
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

class ChatLLM:
    def __init__(self, model_name="llama3.1", db_uri="postgresql://abouzuhayr:@localhost:5432/postgres"):
        # Initialize LLM and database
        self.llm = Ollama(model=model_name)
        self.db = SQLDatabase.from_uri(db_uri)

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
        Answer:"""
                )

        # Create the SQL query chain
        self.write_query = LLMChain(
            llm=self.llm,
            prompt=self.sql_prompt
        )

        # Create the LLM chain
        self.chain = self._create_chain()

    def _create_chain(self):
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
                        result = self.db.run(sql_query)
                    except Exception as e:
                        result = f"Error executing SQL query: {e}"
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

        # Function to add context before generating the final answer
        def add_context(inputs):
            chat_history = self.memory.load_memory_variables({}).get('chat_history', '')
            inputs['chat_history'] = chat_history
            inputs['database_description'] = self.database_description
            return inputs

        add_context_runnable = RunnableLambda(add_context)

        # Combine everything into a chain
        chain = (
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
