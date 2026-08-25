const imageInput = document.getElementById("imageInput");
const uploadArea = document.getElementById("uploadArea");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const removeButton = document.getElementById("removeButton");
const analyzeButton = document.getElementById("analyzeButton");

const loading = document.getElementById("loading");
const results = document.getElementById("results");

const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const probabilities = document.getElementById("probabilities");

let selectedFile = null;


// ========================================
// OPEN FILE SELECTOR
// ========================================

uploadArea.addEventListener("click", () => {
    imageInput.click();
});


// ========================================
// FILE SELECTED
// ========================================

imageInput.addEventListener("change", function () {

    if (this.files.length === 0) {
        return;
    }

    handleFile(this.files[0]);
});


// ========================================
// HANDLE FILE
// ========================================

function handleFile(file) {

    // Check file type
    if (!file.type.startsWith("image/")) {
        alert("Please select a JPG, JPEG, or PNG image.");
        return;
    }

    selectedFile = file;

    // Create preview
    const reader = new FileReader();

    reader.onload = function (event) {

        previewImage.src = event.target.result;

        previewContainer.classList.add("show");

        uploadArea.style.display = "none";

        analyzeButton.disabled = false;

        // Hide previous results
        results.classList.remove("show");
    };

    reader.readAsDataURL(file);
}


// ========================================
// REMOVE IMAGE
// ========================================

removeButton.addEventListener("click", function () {

    selectedFile = null;

    imageInput.value = "";

    previewImage.src = "";

    previewContainer.classList.remove("show");

    uploadArea.style.display = "block";

    analyzeButton.disabled = true;

    results.classList.remove("show");
});


// ========================================
// DRAG & DROP
// ========================================

uploadArea.addEventListener("dragover", function (event) {

    event.preventDefault();

    uploadArea.classList.add("dragover");
});


uploadArea.addEventListener("dragleave", function () {

    uploadArea.classList.remove("dragover");
});


uploadArea.addEventListener("drop", function (event) {

    event.preventDefault();

    uploadArea.classList.remove("dragover");

    const file = event.dataTransfer.files[0];

    if (file) {
        handleFile(file);
    }
});


// ========================================
// ANALYZE IMAGE
// ========================================

analyzeButton.addEventListener("click", async function () {

    if (!selectedFile) {
        alert("Please select an image first.");
        return;
    }


    // Show loading
    loading.classList.add("show");

    results.classList.remove("show");

    analyzeButton.disabled = true;


    // Create form data
    const formData = new FormData();

    formData.append("image", selectedFile);


    try {

        // Send image to Flask
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });


        const data = await response.json();


        // Hide loading
        loading.classList.remove("show");

        analyzeButton.disabled = false;


        // Check for backend error
        if (!response.ok) {

            alert(data.error || "Something went wrong.");

            return;
        }


        // Display prediction
        prediction.textContent = data.prediction;

        confidence.textContent =
            data.confidence.toFixed(2) + "%";


        // Clear old probabilities
        probabilities.innerHTML = "";


        // Create probability bars
        for (const [className, probability] of Object.entries(
            data.probabilities
        )) {

            const row = document.createElement("div");

            row.className = "probability-row";


            row.innerHTML = `
                <div class="probability-info">
                    <span>${className.toUpperCase()}</span>
                    <span>${probability.toFixed(2)}%</span>
                </div>

                <div class="progress">
                    <div
                        class="progress-bar"
                        style="width: ${probability}%"
                    ></div>
                </div>
            `;


            probabilities.appendChild(row);
        }


        // Show results
        results.classList.add("show");


        // Scroll to results
        results.scrollIntoView({
            behavior: "smooth"
        });

    }

    catch (error) {

        loading.classList.remove("show");

        analyzeButton.disabled = false;

        console.error(error);

        alert(
            "Could not connect to the server. " +
            "Make sure the Flask app is running."
        );
    }

});