# TREMORSHIELD updated version

- All 9 tasks are 30 seconds (4 minutes 30 seconds of task time).
- Mouse-only instructions; task start uses a mouse button instead of Space.
- Intro explains the data collected and estimated completion time.
- Normal/Fast/Slow now show a moving blue target to track.
- Finished sessions email both a combined CSV and a ZIP of task CSVs.
- SMTP email sending tries configured port and Gmail SSL port 465 as fallback.
- Added `/admin/test_email?token=...` (POST) to test email configuration.


## Email delivery fix
- Replaced direct Gmail/SMTP delivery with Brevo's HTTPS email API.
- Session ZIP + combined CSV are sent over HTTPS on port 443.
- Added `BREVO_API_KEY`, `BREVO_SENDER_EMAIL`, `BREVO_SENDER_NAME`, and `ADMIN_EMAIL` environment variables.
- `/admin/test_email?token=...` now tests the Brevo API configuration.
