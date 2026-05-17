import os
import pandas as pd
from datetime import datetime


class LabRepository:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def save_report(self, uid, email, data):
        try:
            counts = data.get("counts", {})

            flat_data = {
                "uid": uid,
                "email": email,
                "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "total_tests": sum(counts.values()),
                "normal": counts.get("normal", 0),
                "high": counts.get("high", 0),
                "low": counts.get("low", 0),
                "borderline": counts.get("borderline", 0),
                "overall_interpretation": data.get("overall_interpretation", ""),
                "advice": data.get("advice", ""),
            }

            # ✅ SAVE BIOMARKERS (VALUE + STATUS + UNIT + MEANING)
            categories = data.get("categories", {})
            for cat_name, tests in categories.items():
                if not tests:
                    continue

                for t in tests:
                    # Normalize key
                    base = f"{cat_name}_{t.get('name', '')}".replace(" ", "_").lower()

                    flat_data[f"{base}_value"] = t.get("value")
                    flat_data[f"{base}_status"] = t.get("status")
                    flat_data[f"{base}_unit"] = t.get("unit")
                    flat_data[f"{base}_meaning"] = t.get("meaning")  # ✅ NEW (important)

            new_df = pd.DataFrame([flat_data])

            if os.path.exists(self.file_path):
                existing_df = pd.read_csv(self.file_path)
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                updated_df = new_df

            updated_df.to_csv(self.file_path, index=False)
            return True

        except Exception as e:
            print(f"Storage Error: {e}")
            return False