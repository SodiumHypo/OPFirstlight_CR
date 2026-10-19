@echo off
rem crai pipeline GUI launcher - FirstLight venv has torch + tkinter
rem -I keeps env isolation; workspace root goes on sys.path explicitly
rem Local tool paths live here (published code stays machine-independent)
cd /d "%~dp0..\.."
set "CR_ADB=D:\Software\MuMu Player 12\shell\adb.exe"
set "CR_MUMU_MANAGER=D:\Software\MuMu Player 12\nx_main\MuMuManager.exe"
set "CR_ADB_SERIAL=127.0.0.1:16416"
"firstlight\.venv\Scripts\python.exe" -I -u -c "import sys; sys.path.insert(0, r'%~dp0..\..'); from pipeline.host.gui import main; raise SystemExit(main())"
