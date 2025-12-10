// 2026 Course Site - JavaScript

document.addEventListener('DOMContentLoaded', function() {
  // Load shared header
  loadHeader();

  // Initialize announcement banner
  initAnnouncement();
});

// Load header component
async function loadHeader() {
  try {
    const response = await fetch('header.html');
    if (!response.ok) throw new Error('Header not found');

    const headerHtml = await response.text();
    document.body.insertAdjacentHTML('afterbegin', headerHtml);

    // Set active nav link
    setActiveNav();
  } catch (error) {
    console.error('Failed to load header:', error);
    // Fallback header
    const fallbackHeader = `
      <header>
        <div class="container">
          <h1><a href="index.html">Designing Large Scale AI Systems</a></h1>
          <nav>
            <a href="index.html">Home</a>
            <a href="syllabus.html">Syllabus</a>
            <a href="materials.html">Materials</a>
            <a href="assessments.html">Assessments</a>
            <a href="project.html">Project</a>
            <a href="about.html">About</a>
          </nav>
        </div>
      </header>
    `;
    document.body.insertAdjacentHTML('afterbegin', fallbackHeader);
    setActiveNav();
  }
}

// Highlight current page in navigation
function setActiveNav() {
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  const navLinks = document.querySelectorAll('nav a');

  navLinks.forEach(link => {
    const linkPage = link.getAttribute('href');
    if (linkPage === currentPage) {
      link.classList.add('active');
    }
  });
}

// Initialize announcement banner close functionality
function initAnnouncement() {
  const announcement = document.querySelector('.announcement');
  const closeBtn = document.querySelector('.announcement .close-btn');

  if (announcement && closeBtn) {
    // Check if user has dismissed this announcement
    const dismissed = localStorage.getItem('announcement-dismissed-2026');
    if (dismissed === 'true') {
      announcement.style.display = 'none';
    }

    closeBtn.addEventListener('click', function() {
      announcement.style.display = 'none';
      localStorage.setItem('announcement-dismissed-2026', 'true');
    });
  }
}
