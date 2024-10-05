// Function to switch logo and toggle icon fill color based on color scheme
function switchLogoAndToggleFill() {
    const logo = document.getElementById('flyvis-logo');
    const isDarkMode = document.body.getAttribute('data-md-color-scheme') === 'slate';

    // Switch between dark and light logos
    if (isDarkMode) {
        logo.src = 'images/flyvis_logo_dark@150 ppi.webp';
    } else {
        logo.src = 'images/flyvis_logo_light@150 ppi.webp';
    }

    // Switch the fill color of the toggle button icons
    const toggleIcons = document.querySelectorAll('.md-header__button svg');
    toggleIcons.forEach(icon => {
        if (isDarkMode) {
            icon.style.fill = 'white';  // Change to white in dark mode
        } else {
            icon.style.fill = 'black';  // Change to black in light mode
        }
    });
}

// Initial logo and toggle setup based on the current color scheme
document.addEventListener('DOMContentLoaded', switchLogoAndToggleFill);

// Observe changes to the data-md-color-scheme attribute
const observer = new MutationObserver(switchLogoAndToggleFill);
observer.observe(document.body, {
    attributes: true,
    attributeFilter: ['data-md-color-scheme']
});

// Add event listener to detect system preference changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', switchLogoAndToggleFill);
window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', switchLogoAndToggleFill);


document.addEventListener("DOMContentLoaded", function () {
    // Get the current URL path
    const isHomePage = window.location.pathname === "/" || window.location.pathname === "/index.html";

    // Toggle the logo visibility based on the current page
    if (!isHomePage) {
        document.body.classList.add("show-logo");
    }
});
