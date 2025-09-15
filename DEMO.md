# Sağlık kontrolü
curl http://localhost:8000/health

# TTS isteği
curl -X POST "http://localhost:8000/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text":"Merhaba dünya, bu VibeVoice testidir.", "language":"tr"}' \
  --output output.wav

  curl -X 'POST' \
  'http://localhost:8000/synthesize' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Merhaba dünya, bu bir sesli test mesajıdır.",
  "language": "tr"
}' \
--output test_sentez.wav

echo "Ses dosyası test_sentez.wav olarak kaydedildi."