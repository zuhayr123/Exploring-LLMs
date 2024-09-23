import re
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

class ChatLLM:
    def __init__(self, model_name="llama3.1", db_uri="postgresql://abouzuhayr:@localhost:5432/postgres"):
        # Initialize LLM and database
        self.llm = Ollama(model=model_name)
        self.db = SQLDatabase.from_uri(db_uri)
        
        # Create the SQL query chain
        self.write_query = create_sql_query_chain(llm=self.llm, db=self.db)

        # Prompt template for answering the question
        self.answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

            Question: {question}
            SQL Query: {query}
            SQL Result: {result}
            Answer: """
        )

        # Create the LLM chain
        self.chain = self._create_chain()

    def _create_chain(self):
        # Wrap the SQL query generation and execution logic
        def write_query_with_question(inputs):
            response = self.write_query.invoke(inputs)
            return {'response': response, 'question': inputs['question']}

        write_query_runnable = RunnableLambda(write_query_with_question)

        # Function to extract and execute the SQL query
        def extract_and_execute_sql(inputs):
            response = inputs.get('response', '')
            question = inputs.get('question', '')

            # Regex to find the SQL query
            pattern = re.compile(r'SQLQuery:\s*(.*)')
            match = pattern.search(response)

            if match:
                sql_query = match.group(1).strip()
                result = self.db.run(sql_query)
                return {
                    "question": question,
                    "query": sql_query,
                    "result": result
                }
            else:
                return {
                    "question": question,
                    "query": None,
                    "result": "No SQL query found in the response."
                }

        extract_and_execute = RunnableLambda(extract_and_execute_sql)

        # Combine everything into a chain
        chain = (
            write_query_runnable
            | extract_and_execute
            | self.answer_prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def get_response(self, question):
        database_description = (
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
        # Call the chain with the user question
        print("question" + question)
        response = self.chain.invoke({"question": database_description + "using this info, answer" +  question})
        print("answer" + response)
        return response

# Now you can call this class in your Streamlit app