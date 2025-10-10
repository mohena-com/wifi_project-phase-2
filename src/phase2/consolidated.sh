#!/bin/bash

# ----------- CREDENTIALS SETUP -----------
LMS_COMMON_DIT_USER="LMS_COMMON_DIT"
LMS_COMMON_DIT_PASS="your_password"
LMS_COMMON_DIT_DB="dbinstance"

LMS_SWEEPS_DIT_USER="LMS_SWEEPS_DIT"
LMS_SWEEPS_DIT_PASS="your_password"
LMS_SWEEPS_DIT_DB="dbinstance"

CSIDFCDIT_USER="CSIDFCDIT"
CSIDFCDIT_PASS="your_password"
CSIDFCDIT_DB="dbinstance"

# For 24.1.5.0 (unqualified as _DIT)
LMS_COMMON_USER="LMS_COMMON"
LMS_COMMON_PASS="your_password"
LMS_COMMON_DB="dbinstance"

LMS_SWEEPS_USER="LMS_SWEEPS"
LMS_SWEEPS_PASS="your_password"
LMS_SWEEPS_DB="dbinstance"

check_sqlplus_errors() {
    log="$1"
    errors=$(grep -i "error" "$log" | grep -v "no errors")
    if [[ -z "$errors" ]]; then
        echo "SUCCESS: No errors detected in $log."
    else
        echo "ERROR: Detected errors in $log!"
        exit 1
    fi
}

run_sqlplus() {
    user="$1"
    pass="$2"
    db="$3"
    script="$4"
    log="$5"
    echo "Running $script on $user@$db, logging to $log"
    sqlplus -S "${user}/${pass}@${db}" <<EOF > "$log"
SET AUTOCOMMIT OFF;
@$script
WHENEVER SQLERROR EXIT SQL.SQLCODE
commit;
exit;
EOF
    check_sqlplus_errors "$log"
}

# --- 24.1.2.0 MAIN LMS & SILVER ---
echo "===== Starting 24.1.2.0/main/LMS ====="
cd 24.1.2.0/main/LMS || exit 1
run_sqlplus "$LMS_COMMON_DIT_USER" "$LMS_COMMON_DIT_PASS" "$LMS_COMMON_DIT_DB" "01_main_common.sql" "01_main_common.log"
run_sqlplus "$LMS_SWEEPS_DIT_USER" "$LMS_SWEEPS_DIT_PASS" "$LMS_SWEEPS_DIT_DB" "02_main_sweeps.sql" "02_main_sweeps.log"
cd ../../main/SILVER || exit 1
run_sqlplus "$CSIDFCDIT_USER" "$CSIDFCDIT_PASS" "$CSIDFCDIT_DB" "main_01_silver_cs.sql" "main_01_silver_cs.log"
run_sqlplus "$LMS_COMMON_DIT_USER" "$LMS_COMMON_DIT_PASS" "$LMS_COMMON_DIT_DB" "main_02_silver_common.sql" "main_02_silver_common.log"
cd ../../../

# --- 24.1.3.0 MAIN LMS ---
echo "===== Starting 24.1.3.0/main ====="
cd 24.1.3.0/main || exit 1
run_sqlplus "$LMS_COMMON_DIT_USER" "$LMS_COMMON_DIT_PASS" "$LMS_COMMON_DIT_DB" "01_main_lm_common.sql" "01_main_lm_common.log"
run_sqlplus "$LMS_SWEEPS_DIT_USER" "$LMS_SWEEPS_DIT_PASS" "$LMS_SWEEPS_DIT_DB" "01_main_lm_sweeps.sql" "01_main_lm_sweeps.log"
cd ../../

# --- 24.1.3.1 MAIN LMS ---
echo "===== Starting 24.1.3.1/main ====="
cd 24.1.3.1/main || exit 1
run_sqlplus "$LMS_COMMON_DIT_USER" "$LMS_COMMON_DIT_PASS" "$LMS_COMMON_DIT_DB" "01_main_lm_common.sql" "01_main_lm_common.log"
run_sqlplus "$LMS_SWEEPS_DIT_USER" "$LMS_SWEEPS_DIT_PASS" "$LMS_SWEEPS_DIT_DB" "01_main_lm_sweeps.sql" "01_main_lm_sweeps.log"
cd ../../

# --- 24.1.4.0 MAIN LMS ---
echo "===== Starting 24.1.4.0/main ====="
cd 24.1.4.0/main || exit 1
run_sqlplus "$LMS_COMMON_DIT_USER" "$LMS_COMMON_DIT_PASS" "$LMS_COMMON_DIT_DB" "01_main_lm_common.sql" "01_main_lm_common.log"
cd ../../

# --- 24.1.4.1 MAIN LMS ---
echo "===== Starting 24.1.4.1/main ====="
cd 24.1.4.1/main || exit 1
run_sqlplus "$LMS_COMMON_DIT_USER" "$LMS_COMMON_DIT_PASS" "$LMS_COMMON_DIT_DB" "dml_sit_lm_common_currency_decimal_change.sql" "dml_sit_lm_common_currency_decimal_change.log"
cd ../../

# --- 24.1.5.0 MAIN LMS ---
echo "===== Starting 24.1.5.0/main ====="
cd IDFC_Release/IDFC-LMS-Consolidated_Release/24.1.5.0/main || exit 1
run_sqlplus "$LMS_COMMON_USER" "$LMS_COMMON_PASS" "$LMS_COMMON_DB" "01_main_lm_common.sql" "01_main_lm_common.log"
run_sqlplus "$LMS_SWEEPS_USER" "$LMS_SWEEPS_PASS" "$LMS_SWEEPS_DB" "02_main_lm_sweeps.sql" "02_main_lm_sweeps.log"
cd ../../../../../

echo "All specified releases executed successfully."
