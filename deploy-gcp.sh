#!/bin/bash
# Deploy to Google Cloud Run with Cloud SQL
# Prerequisites: gcloud CLI installed and authenticated

set -e

PROJECT_ID=${1:-$(gcloud config get-value project)}
REGION=${2:-us-central1}
SERVICE_NAME="makearjowork"
CLOUD_SQL_INSTANCE="$PROJECT_ID:$REGION:makearjowork-db"

echo "=== Deploying to Cloud Run (builds from source) ==="
gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --add-cloudsql-instances $CLOUD_SQL_INSTANCE \
    --set-env-vars "USE_CLOUD_SQL=true,CLOUD_SQL_CONNECTION=$CLOUD_SQL_INSTANCE,DB_USER=appuser,DB_NAME=makearjowork,DOMAIN=https://makearjowork.com" \
    --set-secrets "SECRET_KEY=secret-key:latest,DB_PASS=db-password:latest,SMTP_USER=smtp-user:latest,SMTP_PASS=smtp-pass:latest,GEMINI_API_KEY=gemini-api-key:latest"

echo ""
echo "=== Deployment complete! ==="
echo "Your service URL:"
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'

echo ""
echo "=== Next steps ==="
echo "1. Configure Cloudflare DNS to point makearjowork.com to Cloud Run"
