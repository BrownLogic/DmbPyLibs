"""
database.py: provides information to make a database connection
"""

# Following the pattern found here:
# http://stackoverflow.com/questions/23966164/simplest-way-to-have-my-python-program-store-retrieve-information-from-an-online

login_info = {
    "host": "localhost",
    "user": "analysis_user",
    "passwd": "Brown75042",
    "db": "analysis_results",
    "port": 3306}


"""
login_info = {
		"host":"brownlogic.cipbrsumelmh.us-west-2.rds.amazonaws.com",
    "user": "analysis_user",
    "passwd": "Brown75042",
    "db": "analysis_results",
    "port": 3306}
"""