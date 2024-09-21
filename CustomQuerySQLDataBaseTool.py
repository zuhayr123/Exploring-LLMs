from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from sqlalchemy.engine import Result
from typing import Optional, Union, Sequence, Dict, Any
from langchain_core.callbacks import CallbackManagerForToolRun
import re


class CustomQuerySQLDataBaseTool(QuerySQLDataBaseTool):
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Union[str, Sequence[Dict[str, Any]], Result]:
        # Use regex to find the part after "SQLQuery: " and extract the SQL query
        match = re.search(r"SQLQuery:\s*(.*)", query, re.DOTALL)
        
        if match:
            # Extract and clean up the SQL query part
            cleaned_query = match.group(1).strip()
        else:
            # If no match, assume the whole query is the SQL query
            cleaned_query = query.strip("```sql\n").strip("\n```")
        
        # Execute the cleaned SQL query
        return self.db.run_no_throw(cleaned_query)