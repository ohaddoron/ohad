#!/bin/bash

if [[ -z "$1" ]] || [[ -z "$2" ]]  ; then printf "\n\n\tError: Missing parameters:\n\tUsage: `basename $0` <username> <authenticate dbname>\n\n" ; exit 100 ; fi


DB_USER="$1"
DB_NAME="$2"
USER_PASS=`echo ${DB_USER} | md5sum | cut -c4-24`
USER_PASS="MongoDb-${USER_PASS}"
DB_PORT="27017"
DB_HOST="132.66.207.18"

echo ${DB_USER}:${USER_PASS}@${DB_NAME}

printf "\nmongodb://${DB_USER}:${USER_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}\?authSource=admin\n\n"

echo "
use admin

db.createUser(
  {
    user: \"${DB_USER}\",
    pwd:  \"${USER_PASS}\",
        \"roles\" : [
                   { \"role\" : \"readWrite\", \"db\" : \"${DB_NAME}\" } ,
                   { \"role\" : \"dbAdmin\"  , \"db\" : \"${DB_NAME}\" }
                  ]
  }
)


"
