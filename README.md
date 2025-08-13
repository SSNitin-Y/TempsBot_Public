# TempsBot — Is a Weather Bot integrated with AI ("Temps" means weather in French).

Live app: [https://share.streamlit.io/… ](https://weatherai-rpmnd2vdjchnbzggsgbqij.streamlit.app/) <!-- paste your Streamlit URL -->

This repo mirrors the **top-level `app.py`** only, for review/demo.
The full project (APIs, GPT, color engine, etc.) remains private.

# TempsBot 🌦👕

**TempsBot** is an interactive **Streamlit** web app that blends **real-time weather forecasting** with **personalized outfit and skin-care recommendations**.

It uses data from the **OpenWeather API** and **GPT-powered summaries** to:  
- 📍 Show current conditions for your city  
- 📊 Display a **5-day weather forecast** (with temperature, humidity, and condition trends)  
- 👕 Suggest **what to wear today** based on weather, UV index, and wind  
- ☀️ Give **UV & sunscreen advice** tailored to your **Fitzpatrick skin type**  
- 🎨 Recommend **color combinations** for your outfits (with human-readable names)  
- 📄 Let you **download your 5-day plan** as Markdown or PDF  
- 🖤 Support **Dark / Light mode** toggle for better UX  

---

## 🛠️ How It Works
1. **Weather Data** – Fetched from the [OpenWeather One Call API](https://openweathermap.org/api/one-call-api).  
2. **Outfit Logic** – Uses weather & UV thresholds to recommend fabrics, accessories, and styles.  
3. **Sunscreen Advisor** – Adjusts SPF recommendations based on **Fitzpatrick skin type I–VI**.  
4. **Color Suggestions** – Dynamically generated **HSL-based outfit colors** with closest color names.  
5. **GPT Summaries** – Generates friendly, natural-language daily summaries & style tips.  

---

## 🤝 Collaboration
This project was built as a **joint collaboration** between:  
- **[Nitin](https://www.linkedin.com/in/ssny15)**  
- **[Ashraiy](https://www.linkedin.com/in/ashraiy-manohar)**  

We worked together on **feature brainstorming, code refinement, and deployment strategy** to create an easy-to-use weather-fashion assistant.

---

## 🚀 Tech Stack
- **Python** (3.13)  
- **Streamlit** for UI  
- **OpenWeather API** for weather data  
- **OpenAI GPT API** for text generation  
- **Pandas & Matplotlib** for data handling & visualization  
- **ReportLab** for PDF export  
- **BeautifulSoup** for HTML parsing  

---

> Note: This code is **not runnable** on its own. It references modules kept private.
