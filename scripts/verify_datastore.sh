#!/bin/bash

# Configuration variables
PROJECT_ID="jchen-6727"
LOCATION="us"
DATA_STORE_ID="your-datastore-id" # Replace with your actual datastore ID

echo "Fetching schema for Data Store: $DATA_STORE_ID..."

# Execute curl silently (-s) and pipe to jq.
# The 'if .jsonSchema' logic ensures that if the API returns an error message instead of the schema, 
# jq will still pretty-print the error without failing.
curl -s -X GET \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://${LOCATION}-discoveryengine.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/collections/default_collection/dataStores/${DATA_STORE_ID}/schemas/default_schema" \
  | jq 'if .jsonSchema then (.jsonSchema |= fromjson) else . end'