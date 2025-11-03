AI Usage Transparency

As required by the assignment, this document details my use of Google's Gemini AI during this project. I used the AI as a code generation and debugging assistant.

My Role vs. AI's Role

My role was to serve as the architect and project director. I designed the system architecture, defined the technology stack, and specified the logic for each component.

The AI's role was to act as a pair programmer, generating the boilerplate code based on my specifications and helping to refactor code blocks to resolve specific errors that I identified.

Development Process

Architecture Design: I decided on a modular, multi-service architecture using docker-compose to separate concerns. I specified the five services (Database, Ingestor, Aggregator, API, Frontend) and chose the key technologies (PostgreSQL/TimescaleDB, FastAPI, Streamlit) based on the assignment's requirements.

Boilerplate Generation: I instructed the AI to generate the initial files for this architecture, including:

docker-compose.yml

Dockerfile for the frontend and backend

The requirements.txt files for both services

The basic Python scripts (ingestor.py, aggregator.py, api.py, dashboard.py)

The db/init.sql file with the schema I specified.

Debugging and Refinement: I was responsible for the full debugging and integration process. When a service failed, I ran the docker-compose logs command to identify the root cause and traceback. Once I understood the error, I instructed the AI to generate the corrected code snippet.

For example:

API (500 Errors):

I identified an ImportError and had the AI correct the function name from adfullertest to adfuller.

I found a TypeError and instructed the AI to wrap the NumPy boolean in bool() to make it JSON-serializable.

I found a ValueError related to NaN and Infinity values. I directed the AI to generate a fix, which we iterated on (from fillna to df.astype(object).where) until the data was correctly cleaned.

Ingestor (Data Error):

I diagnosed a psycopg2.Error showing a type mismatch. I had the AI generate the code to convert the millisecond timestamp into a Python datetime object.

Aggregator (DB Error):

I identified the cannot run inside a transaction block error and had the AI correct the script to use conn.autocommit = True.

Frontend (Crash):

I found a major NameError caused by an indentation issue and restructured the code.

I found KeyErrors and had the AI add if df.empty: checks to make the dashboard more resilient.

I identified a TypeError with timezones and instructed the AI to use .tz_convert('Asia/Kolkata').

Feature Implementation:

I designed the "Live Stats" section and had the AI generate the st.metric code.

I designed the "File Upload" feature as required by the assignment and had the AI generate the new Streamlit UI and local analytics function.

Documentation:

I provided the outlines for the README.md and this file, and had the AI generate the descriptive text. I also had it generate the architecture.drawio file based on my design.