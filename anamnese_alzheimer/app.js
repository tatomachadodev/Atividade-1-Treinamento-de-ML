const API_URL = 'http://127.0.0.1:8000/api/v1/predict-alzheimer';
const form = document.querySelector('#assessment-form');
const submitButton = document.querySelector('#submit-button');
const errorMessage = document.querySelector('#error-message');
const result = document.querySelector('#result');
const riskBadge = document.querySelector('#risk-badge');
const probability = document.querySelector('#probability');
const recommendation = document.querySelector('#recommendation');
const newAssessment = document.querySelector('#new-assessment');

function getPayload() {
  const payload = {};
  new FormData(form).forEach((value, key) => {
    const field = form.querySelector(`[name="${key}"]:checked`) || form.elements[key];
    const scaleField = form.querySelector(`[name="${key}"][data-model-min]`) || field;
    const modelMin = Number(scaleField.dataset.modelMin);
    const modelMax = Number(scaleField.dataset.modelMax);
    const selectedValue = Number(value);

    if (Number.isFinite(modelMin) && Number.isFinite(modelMax)) {
      payload[key] = modelMin + ((selectedValue - 1) / 4) * (modelMax - modelMin);
    } else {
      payload[key] = selectedValue;
    }
  });
  return payload;
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.hidden = true;
  errorMessage.textContent = '';
}

function showResult(data) {
  const level = data.risk_level.toLowerCase();
  riskBadge.textContent = data.risk_level;
  riskBadge.className = `risk-badge ${level}`;
  probability.textContent = `${(data.probability * 100).toFixed(1).replace('.', ',')}%`;
  recommendation.textContent = data.recommendation;
  result.hidden = false;
  result.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  clearError();
  result.hidden = true;
  submitButton.classList.add('loading');
  submitButton.querySelector('span:first-child').textContent = 'Analisando...';

  const payload = getPayload();
  if (payload.DiastolicBP >= payload.SystolicBP) {
    showError('A pressão sistólica deve ser maior que a diastólica.');
    submitButton.classList.remove('loading');
    submitButton.querySelector('span:first-child').textContent = 'Analisar risco';
    return;
  }

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(data.detail) ? 'Verifique os campos informados.' : data.detail;
      throw new Error(detail || 'Não foi possível processar a avaliação.');
    }
    showResult(data);
  } catch (error) {
    showError(error.message.includes('fetch') ? 'Não foi possível conectar à API. Confirme se o servidor está rodando em http://127.0.0.1:8000.' : error.message);
  } finally {
    submitButton.classList.remove('loading');
    submitButton.querySelector('span:first-child').textContent = 'Analisar risco';
  }
});

newAssessment.addEventListener('click', () => {
  result.hidden = true;
  window.scrollTo({ top: 0, behavior: 'smooth' });
});
