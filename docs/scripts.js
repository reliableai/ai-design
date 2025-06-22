document.addEventListener('DOMContentLoaded', function() {
  // Load and insert the header
  fetch('header.html')
    .then(response => {
      if (!response.ok) {
        throw new Error('Failed to load header');
      }
      return response.text();
    })
    .then(data => {
      // Find the main element
      const mainElement = document.querySelector('main');
      // Insert the header before the main element
      if (mainElement) {
        mainElement.insertAdjacentHTML('beforebegin', data);
      } else {
        document.body.insertAdjacentHTML('afterbegin', data);
      }
      
      // Set active state for current page
      const currentPage = window.location.pathname.split('/').pop() || 'index.html';
      const navLinks = document.querySelectorAll('nav a');
      navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPage) {
          link.classList.add('active');
        }
      });
    })
    .catch(error => {
      console.error('Error loading header:', error);
      // If header fails to load, create a basic header
      const basicHeader = `
        <header>
          <h1>Designing Large Scale AI Systems</h1>
          <nav>
            <ul>
              <li><a href="index.html">Home</a></li>
              <li><a href="schedule.html">Schedule</a></li>
              <li><a href="syllabus.html">Syllabus</a></li>
              <li><a href="assignments.html">Assignments</a></li>
              <li><a href="about.html">About</a></li>
            </ul>
          </nav>
        </header>
      `;
      const mainElement = document.querySelector('main');
      if (mainElement) {
        mainElement.insertAdjacentHTML('beforebegin', basicHeader);
      } else {
        document.body.insertAdjacentHTML('afterbegin', basicHeader);
      }
    });

  // Handle announcement banner if it exists
  const announcementBanner = document.getElementById('announcement');
  if (announcementBanner) {
    const closeBtn = announcementBanner.querySelector('#ann-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', function() {
        announcementBanner.style.display = 'none';
        console.log('Announcement banner hidden.');
      });
    }
  }
});
