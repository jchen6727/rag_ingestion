from google.cloud import resourcemanager_v3
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import discoveryengine_v1 as discoveryengine
from google.api_core.exceptions import GoogleAPICallError, PermissionDenied

def verify_project_context(project_id: str):
    print(f"Validating context and permissions for project: {project_id}...")
    try:
        # A quick, read-only metadata check that requires valid credentials
        rm_client = resourcemanager_v3.ProjectsClient()
        project_info = rm_client.get_project(name=f"projects/{project_id}")
        
        print(f"✓ Context Verified! Project is active. State: {project_info.state}")
        return True
    except GoogleAPICallError as e:
        print(f"❌ Project Context Check Failed. The LRO will abort.")
        print(f"Details: {e}")
        return False


def verify_discovery_engine_api(project_id: str, location: str = "us-central1"):
    print(f"Testing Discovery Engine connection for project: {project_id}...")
    try:
        # 1. Initialize client with regional settings if not global
        client_options = (
            {"api_endpoint": f"{location}-discoveryengine.googleapis.com"}
            if location != "global"
            else None
        )
        client = discoveryengine.DataStoreServiceClient(client_options=client_options)
        
        # 2. Construct parent path to the default collection
        parent = f"projects/{project_id}/locations/{location}/collections/default_collection"
        
        # 3. Trigger a lightweight read operation
        # This checks: API status, billing attachment, and workforce pool permissions
        datastores = client.list_data_stores(parent=parent)
        
        # Pulling the first page forces evaluation of the generator object
        list(datastores) 
        
        print("✓ Success: Discovery Engine API is enabled and accessible.")
        return True

    except PermissionDenied as e:
        print("❌ Authentication Succeeded, but your Identity Pool lacks permission.")
        print("Ensure you have 'roles/discoveryengine.admin' or 'roles/discoveryengine.viewer' on the project.")
        print(f"Details: {e}")
        print(
            "try using:\n",
            "gcloud asset search-all-resources \\",
            f"--project=\"{project_id}\" \\",
            "--asset-types=\"discoveryengine.googleapis.com/DataStore\"",
            sep="\n")
        return False
    except GoogleAPICallError as e:
        print("❌ Discovery Engine API check failed. The API may be disabled or routing is wrong.")
        print(f"Details: {e}")
        print(
            "try using:\n",
            "gcloud asset search-all-resources \\",
            f"--project=\"{project_id}\" \\",
            "--asset-types=\"discoveryengine.googleapis.com/DataStore\"",
            sep="\n")
        return False

# Quick execution test
verify_discovery_engine_api(project_id="jchen-6727")
# Use it in your pipeline script
verify_project_context("jchen-6727")
