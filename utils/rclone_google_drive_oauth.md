# Rclone Google Drive OAuth Client

Go here first: https://console.cloud.google.com/

1. Create/select a project.
2. Enable the Drive API:
   `APIs & Services` -> `Library` -> search `Google Drive API` -> `Enable`.

3. Configure OAuth:
   `Google Auth platform` -> `Get started` / `Branding`
   Use:
   - App name: `rclone`
   - User support email: your email
   - Audience: `External`
   - Contact email: your email

4. Add yourself as a test user:
   `Google Auth platform` -> `Audience` -> `Add users` -> your Gmail.

5. Create the actual credentials:
   `Google Auth platform` -> `Clients` -> `Create client`
   - Application type: `Desktop app`
   - Name: `rclone`
   - Click `Create`

The popup gives you:

```text
Client ID
Client secret
```

Those are the two values to paste into `rclone config` at:

```text
client_id>
client_secret>
```

Do not create an API key. Do not create a service account. You want an OAuth Client ID, application type Desktop app.
