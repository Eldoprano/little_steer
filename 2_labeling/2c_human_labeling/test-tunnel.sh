#!/bin/bash
npm run dev > dev.log 2>&1 &
DEV_PID=$!
sleep 2

npm run share > share.log 2>&1 &
SHARE_PID=$!
sleep 15

URL=$(grep -o 'https://.*\.trycloudflare\.com' share.log | head -n 1)
echo "Found URL: $URL"

if [ -n "$URL" ]; then
  curl -s -v "$URL" > curl.log 2>&1
  echo "Curl exit code: $?"
else
  echo "No URL found in share.log:"
  cat share.log
fi

kill $DEV_PID $SHARE_PID 2>/dev/null
