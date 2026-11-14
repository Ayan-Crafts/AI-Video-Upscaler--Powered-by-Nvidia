; Inno Setup 6 script for AI Upscaler Studio (Windows 10/11 x64).
; Compiled by packaging/windows/build.ps1:
;   ISCC.exe /DAppVersion=1.0.0 /DSourceDir=<pyinstaller onedir> /DOutputDir=<dist\release> installer.iss

#define AppName "AI Upscaler Studio"
#define AppTagline "Powered by NVIDIA"
#define AppCodename "Imaginoromancophilia"
#define AppPublisher "Sanjay Kumar S"
#define AppURL "https://github.com/Ayan-Crafts/Nvidia-AI-Video-Upscaler"
#define AppExeName "AI Upscaler Studio.exe"
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\build\windows\pyinstaller-dist\AI-Upscaler-Studio"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist\release"
#endif

[Setup]
; A new AppId: this product replaces the pre-1.0 "NVIDIA AI Video Studio" dev installer.
AppId={{5B8E2C47-3D1A-4F6B-9C2E-7A1D4E8F0B63}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion} ({#AppCodename})
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
AppComments={#AppName} - {#AppTagline}
VersionInfoVersion={#AppVersion}.0
VersionInfoProductName={#AppName}
VersionInfoDescription={#AppName} Setup
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Per-user install by default (no administrator rights); users may elevate for all users.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename=AI-Upscaler-Studio-v{#AppVersion}-Windows-x64-Setup
SetupIconFile=..\common\icons\ai-upscaler-studio.ico
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName} {#AppVersion}
InfoBeforeFile=installer-info.txt
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=8
CloseApplications=yes
ChangesAssociations=no
ChangesEnvironment=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer.%n%nAI Upscaler Studio bundles Python, PyTorch with the CUDA 12.6 runtime, Qt and FFmpeg. You only need a supported NVIDIA GPU with a current NVIDIA display driver.%n%nAI model weights (about 160 MB) are downloaded automatically the first time they are needed.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Comment: "{#AppName} - {#AppTagline}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon; Comment: "{#AppName} - {#AppTagline}"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
// Informational only: the installer never installs or modifies NVIDIA drivers.
function NvidiaDriverPresent(): Boolean;
begin
  Result := FileExists(ExpandConstant('{sys}\nvml.dll')) or
            FileExists(ExpandConstant('{sys}\nvidia-smi.exe')) or
            FileExists(ExpandConstant('{sys}\nvcuda.dll'));
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = wpReady) and (not WizardSilent()) and (not NvidiaDriverPresent()) then
    MsgBox('No NVIDIA display driver was detected on this computer.' + #13#10 + #13#10 +
           'AI Upscaler Studio can be installed, but AI interpolation and upscaling need a ' +
           'supported NVIDIA GPU with a current NVIDIA driver.' + #13#10 + #13#10 +
           'Download drivers only from NVIDIA: https://www.nvidia.com/en-us/drivers/',
           mbInformation, MB_OK);
end;
