#!/bin/bash

if [[ -z $1 ]] ; then echo "Error: Missing username" ; exit 1 ; fi

ABBR="MongoDb-"

USER_NAME="$1"

PASS=`echo ${USER_NAME} | md5sum | cut -c4-24`
PASS="${ABBR}${PASS}"


printf "\n\n\tSuggested password for ${USER_NAME} is : ${PASS}\n\n"
