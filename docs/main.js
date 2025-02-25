document.addEventListener("DOMContentLoaded", function () {
    // Populate the Schedule Section dynamically with structured data
    const scheduleData = [
      {
        date: "2025-01-10",
        lecture: "Introduction to AI Systems",
        topic: "Course overview and introduction",
        slides: "slides/lecture1.pdf",
        notes: "notes/lecture1.html",
        deadline: "HW1 released"
      },
      {
        date: "2025-01-17",
        lecture: "Scalability in AI",
        topic: "Distributed systems basics",
        slides: "slides/lecture2.pdf",
        notes: "notes/lecture2.html",
        deadline: "HW1 due"
      },
      {
        date: "2025-01-24",
        lecture: "Distributed Systems Deep Dive",
        topic: "In-depth discussion on distributed architectures",
        slides: "slides/lecture3.pdf",
        notes: "notes/lecture3.html",
        deadline: "Assignment 1 posted"
      }
      // Add more entries as needed
    ];
  
    const tbody = document.getElementById("schedule-body");
    scheduleData.forEach(item => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${item.date}</td>
        <td>${item.lecture}</td>
        <td>${item.topic}</td>
        <td>${item.slides ? `<a href="${item.slides}" target="_blank">Slides</a>` : ""}</td>
        <td>${item.notes ? `<a href="${item.notes}" target="_blank">Notes</a>` : ""}</td>
        <td>${item.deadline}</td>
      `;
      tbody.appendChild(row);
    });
  });
  