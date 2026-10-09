notes on logging in:

if missing the proper config file:
```json
{
  "universe_domain": "googleapis.com",
  "universe_cloud_web_domain": "cloud.google",
  "type": "external_account_authorized_user_login_config",
  "audience": "//iam.googleapis.com/locations/global/workforcePools/suny-wfif-pool-glb/providers/suny-wfif-pvdr-glb",
  "auth_url": "https://auth.cloud.google/authorize",
  "token_url": "https://sts.googleapis.com/v1/oauthtoken",
  "token_info_url": "https://sts.googleapis.com/v1/introspect"
}
```
can generate with command

```sh
gcloud iam workforce-pools create-login-config locations/global/workforcePools/suny-wfif-pool-glb/providers/suny-wfif-pvdr-glb --output-file="gcloud.json"
```

and move into .auth/ (or some other) personal directory

for ADC, need to remove old quota project

```sh
gcloud auth application-default login --disable-quota-project --login-config=".auth/gcloud.json"
```

then set to the new quota project & billing
```sh
gcloud config set project brk-prj-salvador-dura-bern-sbx
gcloud config set billing/quota_project brk-prj-salvador-dura-bern-sbx
```

afterwards, the gcloud CLI and app ADC authentication commands.

```sh
gcloud auth login --login-config=".auth/gcloud.json"
gcloud auth application-default login --login-config=".auth/gcloud.json"
```

after setting up project, recommended checks (ensure billed to right project):

use `gcloud config list` with `core/project` or `billing/quota_project` to check projects 

```sh
gcloud config list core/project                      
[core]
project = brk-prj-salvador-dura-bern-sbx

Your active configuration is: [default]
```

```sh
gcloud config list billing/quota_project
[billing]
quota_project = brk-prj-salvador-dura-bern-sbx

Your active configuration is: [default]
```

check the ADC .json with `cat` (some text truncated to ...), and ensure `quota_project_id` is correct...

```sh
cat ~/.config/gcloud/application_default_credentials.json
{
  "audience": "//iam.googleapis.com/locations/global/workforcePools/suny-wfif-pool-glb/providers/suny-wfif-pvdr-glb",
  "client_id": ...
  "client_secret": ...
  "quota_project_id": "brk-prj-salvador-dura-bern-sbx",
  "refresh_token": ...
  "token_info_url": "https://sts.googleapis.com/v1/introspect",
  "token_url": "https://sts.googleapis.com/v1/oauthtoken",
  "type": "external_account_authorized_user",
  "universe_domain": "googleapis.com"
}
```

and validate that it goes through to the project through `gcloud beta billing ...`

```sh
gcloud beta billing projects describe $(gcloud config get-value project)
```