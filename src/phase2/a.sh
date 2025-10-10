#!/bin/bash

# Edit these credentials before running, or export as environment variables:
LMS_COMMON_DIT_USER="LMS_COMMON_DIT"
LMS_COMMON_DIT_PASS="your_password"
LMS_COMMON_DIT_DB="dbinstance"

LMS_SWEEPS_DIT_USER="LMS_SWEEPS_DIT"
LMS_SWEEPS_DIT_PASS="your_password"
LMS_SWEEPS_DIT_DB="dbinstance"

CSIDFCDIT_USER="CSIDFCDIT"
CSIDFCDIT_PASS="your_password"
CSIDFCDIT_DB="dbinstance"

# Log and check function
run_sqlplus() {
    user=$1
    pass=$2
    db=$3
    script=$4
    log=$5

    echo "Running $script on $user@$db, logging to $log"
    sqlplus -S "${user}/${pass}@${db}" <<EOF > "$log"
SET AUTOCOMMIT OFF;
@$script
WHENEVER SQLERROR EXIT SQL.SQLCODE
commit;
exit;
EOF

    # Check for errors
    errors=$(grep -i "error" "$log" | grep -v "no errors")
    if [[ -z "$errors" ]]; then
        echo "SUCCESS: No errors detected in $log."
    else
        echo "ERROR: Detected errors in $log!"
        exit 1
    fi
}

# -- LMS release steps
cd 24.1.2.0/main/LMS || exit 1
run_sqlplus "$LMS_COMMON_DIT_USER" "$LMS_COMMON_DIT_PASS" "$LMS_COMMON_DIT_DB" "01_main_common.sql" "01_main_common.log"
run_sqlplus "$LMS_SWEEPS_DIT_USER" "$LMS_SWEEPS_DIT_PASS" "$LMS_SWEEPS_DIT_DB" "02_main_sweeps.sql" "02_main_sweeps.log"

cd ../../main/SILVER || exit 1
run_sqlplus "$CSIDFCDIT_USER" "$CSIDFCDIT_PASS" "$CSIDFCDIT_DB" "main_01_silver_cs.sql" "main_01_silver_cs.log"
run_sqlplus "$LMS_COMMON_DIT_USER" "$LMS_COMMON_DIT_PASS" "$LMS_COMMON_DIT_DB" "main_02_silver_common.sql" "main_02_silver_common.log"

echo "Release scripts executed for all schemas and folders successfully."

