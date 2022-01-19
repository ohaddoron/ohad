#!/bin/bash

if [[ -z "$1" ]] || [[ -z "$2" ]]  ; then printf "\n\n\tError: Missing parameters:\n\tUsage: `basename $0` <username> <authenticate dbname>\n\n" ; exit 100 ; fi


DB_USER="$1"
DB_NAME="$2"

echo "
use admin


db.grantRolesToUser(
	{
		user: \"${DB_USER}\",
		roles: [
			{\"role\" : \"readWrite\", \"db\" : \"${DB_NAME}\"}
		]
	}
)

"
