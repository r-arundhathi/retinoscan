const uploadBox = document.getElementById("uploadBox");
const fileInput = document.getElementById("fileInput");
const uploadPrompt = document.getElementById("uploadPrompt");
const previewImg = document.getElementById("previewImg");
const loading = document.getElementById("loading");
const resultCard = document.getElementById("resultCard");
const heatmapSection = document.getElementById("heatmapSection");
const origImg = document.getElementById("origImg");
const heatImg = document.getElementById("heatImg");
const downloadBtn = document.getElementById("downloadBtn");

uploadBox.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewImg.style.display = "block";
        uploadPrompt.style.display = "none";
    };
    reader.readAsDataURL(file);

    runPrediction(file);
});

function runPrediction(file) {
    resultCard.style.display = "none";
    heatmapSection.style.display = "none";
    downloadBtn.style.display = "none";
    loading.style.display = "block";

    const formData = new FormData();
    formData.append("image", file);

    fetch("/predict", { method: "POST", body: formData })
        .then((res) => res.json())
        .then((data) => {
            loading.style.display = "none";
            renderResult(data);
        })
        .catch((err) => {
            loading.style.display = "none";
            alert("Error analyzing image: " + err);
        });
}

function renderResult(data) {
    resultCard.style.display = "block";
    resultCard.className = "result-card";
    resetBtn.style.display = "block";

    if (data.is_uncertain) {
        resultCard.classList.add("result-uncertain");
        resultCard.innerHTML = `
            <div class="result-icon">🤔</div>
            <div class="result-label" style="color:#f1c40f;">Unable to Determine</div>
            <div class="result-confidence">This image doesn't clearly match a recognized pattern</div>
        `;
    } else if (data.is_normal) {
        resultCard.classList.add("result-normal");
        resultCard.innerHTML = `
            <div class="result-icon">✅</div>
            <div class="result-label" style="color:#2ecc71;">Looks Normal</div>
            <div class="result-confidence">No significant signs detected</div>
        `;
    } else {
        resultCard.classList.add("result-disease");
        resultCard.innerHTML = `
            <div class="result-icon">${data.icon}</div>
            <div class="result-label" style="color:${data.color};">${data.label}</div>
            <div class="result-confidence">${data.confidence}% confidence</div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width:${data.confidence}%; background-color:${data.color};"></div>
            </div>
        `;
    }

    origImg.src = "data:image/png;base64," + data.original_image;
    heatImg.src = "data:image/png;base64," + data.heatmap_image;
    heatmapSection.style.display = "block";
    downloadBtn.style.display = "block";
}

resetBtn.addEventListener("click", () => {
    fileInput.value = "";
    previewImg.src = "";
    previewImg.style.display = "none";
    uploadPrompt.style.display = "block";
    resultCard.style.display = "none";
    heatmapSection.style.display = "none";
    downloadBtn.style.display = "none";
    resetBtn.style.display = "none";
});