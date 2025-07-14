with open(".env", "w", encoding="utf-8") as f:
    f.write("ARIZE_API_KEY=your_actual_api_key\n")
    f.write("ARIZE_SPACE_ID=your_actual_space_id\n")

print(" .env file created with correct UTF-8 encoding.")