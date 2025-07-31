// Reusable Toast Function
function showToast(message, type = 'info', delay = 3500) {
    // Ensure toast container exists
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        toastContainer.style.zIndex = '1100';
        document.body.appendChild(toastContainer);
    }

    const toastId = 'toast-' + Date.now();
    let iconHtml = '';
    let title = '提示';
    
    switch (type) {
        case 'success':
            iconHtml = '<i class="bi bi-check-circle-fill text-success me-2"></i>';
            title = '成功';
            break;
        case 'danger':
            iconHtml = '<i class="bi bi-exclamation-triangle-fill text-danger me-2"></i>';
            title = '错误';
            break;
        case 'info':
        default:
            iconHtml = '<i class="bi bi-info-circle-fill text-info me-2"></i>';
            title = '信息';
            break;
    }

    const toastHTML = `
        <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true" data-bs-delay="${delay}">
            <div class="toast-header">
                ${iconHtml}
                <strong class="me-auto">${title}</strong>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>`;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHTML);
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { delay: delay });
    
    // Custom styling for toast header and body to match the theme
    const toastHeader = toastElement.querySelector('.toast-header');
    const toastBody = toastElement.querySelector('.toast-body');
    
    toastHeader.style.backgroundColor = 'var(--c-card)';
    toastBody.style.backgroundColor = 'var(--c-card)';
    toastHeader.style.color = 'var(--c-text)';
    toastBody.style.color = 'var(--c-text)';
    toastHeader.style.borderBottomColor = 'var(--c-border)';


    toastElement.addEventListener('hidden.bs.toast', () => toastElement.remove());
    
    toast.show();
} 