document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultDiv = document.getElementById('predictionResult');
    const loadingDiv = document.getElementById('loading');
    const resultContent = document.getElementById('resultContent');

    // Intersection Observer for form sections
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    });

    // Observe all form sections
    document.querySelectorAll('.form-section').forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        observer.observe(section);
    });

    // Function to make API request with retry logic
    async function makeRequest(data, retries = 3) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

        try {
            const response = await fetch('/api/verify_engagement', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            clearTimeout(timeoutId);

            if (error.name === 'AbortError') {
                throw new Error('Request timed out. Please try again.');
            }

            if (retries > 0 && (error.name === 'TypeError' || error.message.includes('Failed to fetch'))) {
                console.log(`Retrying... ${retries} attempts left`);
                await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second before retry
                return makeRequest(data, retries - 1);
            }

            throw error;
        }
    }

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading animation
        loadingDiv.style.display = 'block';
        resultDiv.style.display = 'none';
        
        // Collect form data
        const formData = {
            absence_reason_code: parseInt(document.getElementById('absence_reason_code').value),
            absence_month: parseInt(document.getElementById('absence_month').value),
            weekday_code: parseInt(document.getElementById('weekday_code').value),
            season_indicator: parseInt(document.getElementById('season_indicator').value),
            commute_cost: parseFloat(document.getElementById('commute_cost').value),
            commute_distance: parseFloat(document.getElementById('commute_distance').value),
            years_at_company: parseFloat(document.getElementById('years_at_company').value),
            daily_workload: parseFloat(document.getElementById('daily_workload').value),
            performance_target: parseFloat(document.getElementById('performance_target').value),
            disciplinary_action: parseInt(document.getElementById('disciplinary_action').value),
            education_level: parseInt(document.getElementById('education_level').value),
            alcohol_consumption: parseInt(document.getElementById('alcohol_consumption').value),
            tobacco_use: parseInt(document.getElementById('tobacco_use').value),
            body_weight_kg: parseFloat(document.getElementById('body_weight_kg').value),
            body_height_cm: parseFloat(document.getElementById('body_height_cm').value),
            bmi_score: parseFloat(document.getElementById('bmi_score').value)
        };

        try {
            const result = await makeRequest(formData);
            
            // Hide loading animation
            loadingDiv.style.display = 'none';
            
            // Show result
            resultDiv.style.display = 'block';
            resultDiv.classList.add('show');
            
            if (result.error) {
                resultDiv.className = 'prediction-result alert alert-danger show';
                resultContent.innerHTML = `
                    <div class="result-item">
                        <p><strong>Error:</strong> ${result.error}</p>
                    </div>
                `;
            } else {
                resultDiv.className = 'prediction-result alert alert-success show';
                resultContent.innerHTML = `
                    <div class="result-item">
                        <p><strong>Predicted Class:</strong> ${result.class_description}</p>
                    </div>
                    <div class="result-item">
                        <p><strong>Confidence Score:</strong> ${(result.confidence_score * 100).toFixed(2)}%</p>
                    </div>
                `;
            }
        } catch (error) {
            loadingDiv.style.display = 'none';
            resultDiv.style.display = 'block';
            resultDiv.className = 'prediction-result alert alert-danger show';
            
            let errorMessage = 'An error occurred while making the prediction. ';
            if (error.message.includes('timed out')) {
                errorMessage += 'The request took too long to complete. ';
            } else if (error.message.includes('Failed to fetch')) {
                errorMessage += 'Unable to connect to the server. ';
            }
            errorMessage += 'Please try again.';
            
            resultContent.innerHTML = `
                <div class="result-item">
                    <p><strong>Error:</strong> ${errorMessage}</p>
                </div>
            `;
            console.error('Prediction error:', error);
        }
    });
});
