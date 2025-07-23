:: start-backend.bat
@echo off
:: Activate the virtual environment
call venv\Scripts\activate

:: Run uvicorn with the specified app and reload option
uvicorn app.main:app --reload

:: Deactivate the virtual environment when done (optional, comment out if not needed)
:: deactivate