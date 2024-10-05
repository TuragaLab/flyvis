// // Function to switch logo and toggle icon fill color based on color scheme
// function switchLogoAndToggleFill() {
//     const logo1 = document.getElementById('flyvis-logo');
//     const logo2 = document.getElementById('_flyvis-logo');

//     // Ensure logos exist before continuing
//     if (!logo1 || !logo2) return;

//     const isDarkMode = document.body.getAttribute('data-md-color-scheme') === 'slate';

//     // Use an absolute path to the logo
//     const baseUrl = window.location.origin;  // Get the site base URL dynamically

//     // Construct logo path based on whether it's the homepage or not
//     const logoPathLight = `${baseUrl}/images/flyvis_logo_light@150%20ppi.webp`;
//     const logoPathDark = `${baseUrl}/images/flyvis_logo_dark@150%20ppi.webp`;

//     // Switch between dark and light logos
//     const logoSrc = isDarkMode ? logoPathDark : logoPathLight;
//     logo1.src = logoSrc;
//     logo2.src = logoSrc;

//     // Switch the fill color of the toggle button icons
//     const toggleIcons = document.querySelectorAll('.md-header__button svg');
//     toggleIcons.forEach(icon => {
//         icon.style.fill = isDarkMode ? 'white' : 'black';
//     });
// }

// // Initial logo and toggle setup based on the current color scheme
// document.addEventListener('DOMContentLoaded', switchLogoAndToggleFill);

// // Observe changes to the data-md-color-scheme attribute
// const observer = new MutationObserver(switchLogoAndToggleFill);
// observer.observe(document.body, {
//     attributes: true,
//     attributeFilter: ['data-md-color-scheme']
// });

// // Add a single event listener for system preference changes
// window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', switchLogoAndToggleFill);
