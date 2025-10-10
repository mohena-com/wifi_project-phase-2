#!/bin/bash

# Set these environment variables as needed before running
# export ORACLE_USER_LMSCOMMON=user
# export ORACLE_PASSWORD_LMSCOMMON=pass
# export ORACLE_DB_LMSCOMMON=DBNAME

# Add additional user/password/db variables for all other schemas as needed

release_folders=(
  "24.1.2.0mainLMS"
  "24.1.2.0mainSILVER"
  "24.1.3.0main"
  "24.1.3.1main"
  "24.1.4.0main"
  "24.1.4.1main"
  "IDFCReleaseIDFC-LMS-ConsolidatedRelease24.1.5.0main"
  "24.2.0.013JANTAGK1K1LMS-DBSCRIPTSdbScriptsOracle"
  "24.2.0.001-cs"
  "24.2.0.002-lmcommon"
  "24.2.0.0main03-sweep"
  "24.2.0.0main04-rpt"
  "24.2.1.0main"
  "24.2.2.0"
)

# Define scripts for each folder and schema in associative arrays
declare -A scripts

# Examples for some releases, add others similarly:
scripts["24.1.2.0mainLMS"]=("LMSCOMMONDIT" "01maincommon.sql" "LMSSWEEPSDIT" "02mainsweeps.sql")
scripts["24.1.2.0mainSILVER"]=("CSIDFCDIT" "main01silvercs.sql" "LMSCOMMONDIT" "main02silvercommon.sql")
scripts["24.1.3.0main"]=("LMSCOMMONDIT" "01mainlmcommon.sql" "LMSSWEEPSDIT" "01mainlmsweeps.sql")
# Add other folders/scripts as per the release notes

run_sql_script() {
  local user="$1"
  local password="$2"
  local db="$3"
  local script="$4"
  sqlplus -S "${user}/${password}@${db}" <<EOF
SET AUTOCOMMIT OFF
@${script}
WHENEVER SQLERROR EXIT SQL.SQLCODE
commit;
exit;
EOF
}

for folder in "${release_folders[@]}"; do
  echo "Processing release folder: $folder"
  cd "$folder" || { echo "Folder $folder not found."; continue; }

  # Get scripts and schemas for this folder
  s=("${scripts[$folder]}")
  for ((i=0; i<${#s[@]}; i+=2)); do
    schema="${s[$i]}"
    script="${s[$i+1]}"

    # Customize user, password, db per schema (may need mappings)
    user_var="ORACLE_USER_${schema}"
    pass_var="ORACLE_PASSWORD_${schema}"
    db_var="ORACLE_DB_${schema}"

    user="${!user_var}"
    password="${!pass_var}"
    db="${!db_var}"

    echo "Running script $script on schema $schema"
    run_sql_script "$user" "$password" "$db" "$script"
  done

  cd ..
done

echo "All releases processed."

