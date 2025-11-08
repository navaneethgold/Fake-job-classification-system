const form = document.getElementById("jobForm");
const resultDiv = document.getElementById("result");
const labelP = document.getElementById("label");
const probP = document.getElementById("probability");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const data = {
    text: document.getElementById("text").value,
    employment_type: document.getElementById("employment_type").value,
    required_experience: document.getElementById("required_experience").value,
    required_education: document.getElementById("required_education").value,
    telecommuting: document.getElementById("telecommuting").checked ? 1 : 0,
    has_company_logo: document.getElementById("has_company_logo").checked ? 1 : 0,
    has_questions: document.getElementById("has_questions").checked ? 1 : 0,
  };

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error("Prediction failed");
    }

    const result = await response.json();

    resultDiv.classList.remove("hidden");
    labelP.textContent =
      result.label === "Fake"
        ? "⚠️ Fake Job Posting Detected"
        : "✅ Legitimate Job Posting";
    labelP.style.color = result.label === "Fake" ? "#dc2626" : "#16a34a";

    probP.textContent = `Probability (Fake): ${(result.proba_fake * 100).toFixed(2)}%`;
  } catch (error) {
    console.error(error);
    resultDiv.classList.remove("hidden");
    labelP.textContent = "❌ Error: Unable to predict";
    probP.textContent = "";
    labelP.style.color = "#dc2626";
  }
});
