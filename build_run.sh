#!/bin/bash
######################################################################################
# usage:    ./build_run.sh --name <UNIQUE_PROJECT_NAME> --svc-acc-email <SERVICE_ACCOUNT_EMAIL>
######################################################################################
set -e
while test -n "$1"; do
    case "$1" in
        --name)
            NAME=$2
            shift
            ;;
        --svc-acc-email)
            SVC_ACC=$2
            shift
            ;;
       *)
            echo "Unknown argument: $1"
            ;;
    esac
    shift
done



function cloud_build_submit () {
    NAME=${NAME}
    [[ -z "$NAME" ]] && echo "Variable name: ${NAME} is empty" && exit 1
    SVC_ACC=${SVC_ACC:=""}
    [[ -z "$SVC_ACC" ]] && echo "Variable name: ${SVC_ACC} is empty" && exit 1
    PROJECT=$(gcloud config get-value project)
    gcloud builds submit . \
        --config cloudbuild.yaml \
        --substitutions _SERVICE_NAME=$NAME \
        --gcs-source-staging-dir gs://artifacts.$PROJECT.appspot.com/source \
        --impersonate-service-account=$SVC_ACC
}


main () {
    cloud_build_submit
}


main