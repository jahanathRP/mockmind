document.addEventListener('DOMContentLoaded', () => {
    // Basic interaction for alerts
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.transition = 'opacity 0.5s ease';
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 500);
        }, 5000); // Remove after 5 seconds
    });
});
