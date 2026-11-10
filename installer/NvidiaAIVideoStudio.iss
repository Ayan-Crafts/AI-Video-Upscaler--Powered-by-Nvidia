#define MyAppName "NVIDIA AI Video Studio"
#define MyAppVersion "0.1.0-dev"
#define MyAppExeName "NVIDIA-AI-Video-Studio.exe"
[Setup]
AppId={{A25F069B-9C40-4B6B-949B-19B7B9C050E2}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\NVIDIA AI Video Studio
DefaultGroupName={#MyAppName}
OutputDir=..\dist-installer
OutputBaseFilename=NVIDIA-AI-Video-Studio-Setup-{#MyAppVersion}
ArchitecturesInstallIn64BitMode=x64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
[Files]
Source: "..\dist\NVIDIA-AI-Video-Studio\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: postinstall nowait skipifsilent
