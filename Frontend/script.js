const form = document.getElementById("jobForm");
const resultDiv = document.getElementById("result");
const labelP = document.getElementById("label");
const probP = document.getElementById("probability");
const resultIcon = document.getElementById("resultIcon");
const submitBtn = document.getElementById("submitBtn");
const clearBtn = document.getElementById("clearBtn");
const explainBtn = document.getElementById("explainBtn");
let lastPayload = null;  
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultDiv.classList.add("hidden");

  submitBtn.disabled = true;
  submitBtn.querySelector("span").textContent = "Checking...";

  const data = {
    text: document.getElementById("text").value,
    employment_type: document.getElementById("employment_type").value,
    required_experience: document.getElementById("required_experience").value,
    required_education: document.getElementById("required_education").value,
    telecommuting: document.getElementById("telecommuting").checked ? 1 : 0,
    has_company_logo: document.getElementById("has_company_logo").checked ? 1 : 0,
    has_questions: document.getElementById("has_questions").checked ? 1 : 0,
  };
  lastPayload = data;
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    if (!response.ok) throw new Error("Prediction failed");

    const result = await response.json();

    // show result
    resultDiv.classList.remove("hidden");

    if (result.label === "Fake") {
      labelP.textContent = "Fake Job Posting Detected";
      labelP.style.color = "#b91c1c"; // danger
      resultIcon.innerHTML = '<img src="icon.png" alt="icon" style="width:4rem; background-color:white;">';
    } else {
      labelP.textContent = "Likely Legitimate Job Posting";
      labelP.style.color = "#047857"; // success
      resultIcon.innerHTML = '<img src="icon.png" alt="icon" style="width:4rem; background-color:white;">';
    }

    probP.textContent = `Probability (Fake): ${(result.proba_fake * 100).toFixed(2)}%`;
  } catch (err) {
    resultDiv.classList.remove("hidden");
    labelP.textContent = "Error — Unable to get prediction";
    labelP.style.color = "#b91c1c";
    probP.textContent = "";
    resultIcon.innerHTML = '<img src="icon.png" alt="icon" style="width:4rem; background-color:white;">';
    console.error(err);
  } finally {
    submitBtn.disabled = false;
    submitBtn.querySelector("span").textContent = "Check Job";
  }
});

clearBtn.addEventListener("click", () => {
  document.getElementById("text").value = "";
  document.getElementById("employment_type").value = "Unknown";
  document.getElementById("required_experience").value = "Unknown";
  document.getElementById("required_education").value = "Unknown";
  document.getElementById("telecommuting").checked = false;
  document.getElementById("has_company_logo").checked = false;
  document.getElementById("has_questions").checked = false;
  resultDiv.classList.add("hidden");
  labelP.textContent = "—";
  probP.textContent = "Probability (Fake): —";
  resultIcon.innerHTML = "";
});
explainBtn.addEventListener("click", async () => {
  // ensure we have a payload to explain
  if (!lastPayload) {
    alert("No prediction available to explain. Please run a prediction first.");
    return;
  }

  explainBtn.disabled = true;
  explainBtn.textContent = "Explaining...";

  try {
    // Call explain endpoint (top_n can be adjusted)
    const topN = 10;
    const resp = await fetch(`/explain?top_n=${topN}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(lastPayload),
    });

    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`Explain failed: ${resp.status} ${txt}`);
    }

    const explanation = await resp.json();
    // explanation = { bias:..., positive:[{feature,contrib},...], negative:[...] }

    // Render explanation into result panel (create if not exists)
    let explainContainer = document.getElementById("explainContainer");
    if (!explainContainer) {
      explainContainer = document.createElement("div");
      explainContainer.id = "explainContainer";
      explainContainer.style.marginTop = "12px";
      explainContainer.style.paddingTop = "12px";
      explainContainer.style.borderTop = "1px solid rgba(15,23,42,0.04)";
      // place it inside the result panel
      document.getElementById("result").appendChild(explainContainer);
    }

    // Build HTML for positive & negative lists
    const pos = explanation.positive || [];
    const neg = explanation.negative || [];
    const bias = typeof explanation.bias !== "undefined" ? explanation.bias : null;

    const makeListHTML = (items, color) => {
      if (!items.length) return `<p style="color:${color};margin:0.35rem 0 1rem">No features</p>`;
      return `<ol style="padding-left:18px; margin:0.25rem 0;">
        ${items.map(item =>
          `<li style="margin:6px 0;">
             <span style="font-weight:600">${escapeHtml(item.feature)}</span>
             <span style="color:#6b7280"> — ${Number(item.contrib).toFixed(4)}</span>
           </li>`).join("")}
      </ol>`;
    };

    explainContainer.innerHTML = `
      <div style="display:flex; gap:18px; flex-wrap:wrap;">
        <div style="flex:1; min-width:220px;">
          <h4 style="margin:0 0 8px 0; color:#16a34a;">Top positive contributors (push → Fake)</h4>
          ${makeListHTML(pos, "#16a34a")}
        </div>
        <div style="flex:1; min-width:220px;">
          <h4 style="margin:0 0 8px 0; color:#ef4444;">Top negative contributors (push → Real)</h4>
          ${makeListHTML(neg, "#ef4444")}
        </div>
      </div>
      <div style="margin-top:10px; color:#6b7280; font-size:13px;">
        ${ bias !== null ? `Base bias (model intercept): ${Number(bias).toFixed(4)}` : "" }
      </div>
    `;

    // ensure the result area is visible
    document.getElementById("result").classList.remove("hidden");
  } catch (err) {
    console.error(err);
    alert("Explain failed: " + (err.message || err));
  } finally {
    explainBtn.disabled = false;
    explainBtn.textContent = "Explain";
  }
});

// ---------- small helper to escape HTML ----------
function escapeHtml(str) {
  if (typeof str !== "string") return str;
  return str.replace(/[&<>"'`=\/]/g, function (s) {
    return ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
      '/': '&#x2F;',
      '`': '&#x60;',
      '=': '&#x3D;'
    })[s];
  });
}