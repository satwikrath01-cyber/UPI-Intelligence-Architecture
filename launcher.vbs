Dim objShell, strDir
strDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
Set objShell = WScript.CreateObject("WScript.Shell")
objShell.CurrentDirectory = strDir
objShell.Run "cmd /c python -m streamlit run app.py --server.headless true >> streamlit.log 2>&1", 0, False
Set objShell = Nothing
