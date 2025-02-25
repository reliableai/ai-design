document.addEventListener('DOMContentLoaded', function() {
  const announcementBanner = document.getElementById('announcement');
  if (!announcementBanner) {
    console.error('Announcement banner not found.');
    return;
  }
  
  const closeBtn = announcementBanner.querySelector('#ann-close');
  if (!closeBtn) {
    console.error('Close button (#ann-close) not found within the announcement banner.');
    return;
  }
  
  closeBtn.addEventListener('click', function() {
    announcementBanner.style.display = 'none';
    console.log('Announcement banner hidden.');
  });
});
