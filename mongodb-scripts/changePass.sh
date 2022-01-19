#!/bin/bash

if [[ -z $1 ]] ; then echo "Error: Missing username" ; exit 1 ; fi

ABBR="MongoDb-"

USER_NAME="$1"

PASS=`echo ${USER_NAME} | md5sum | cut -c4-24`
PASS="${ABBR}${PASS}"

printf "\n\nuse admin\ndb.changeUserPassword('${USER_NAME}', '${PASS}')\n"

