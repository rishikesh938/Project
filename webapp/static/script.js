document.getElementById("weatherForm").addEventListener("submit", function(event) {
    event.preventDefault();

    const data = {
        temperature: document.getElementById("temperature").value,
        humidity: document.getElementById("humidity").value,
        cloud_cover: document.getElementById("cloud_cover").value,
        wind_speed: document.getElementById("wind_speed").value,
        precipitation: document.getElementById("precipitation").value
    };

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("predictionResult").textContent = data.prediction;
    })
    .catch(error => console.error("Error:", error));
});
