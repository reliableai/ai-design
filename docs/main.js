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

  const container = document.getElementById("schedule-container");
  scheduleData.forEach(item => {
    const card = document.createElement("div");
    card.className = "schedule-card";
    card.innerHTML = `
      <h3>${item.date} - ${item.lecture}</h3>
      <div class="schedule-field"><strong>Topic:</strong> ${item.topic}</div>
      <div class="schedule-field">
        <strong>Slides:</strong> ${item.slides ? `<a href="${item.slides}" target="_blank">View</a>` : "N/A"}
      </div>
      <div class="schedule-field">
        <strong>Notes:</strong> ${item.notes ? `<a href="${item.notes}" target="_blank">View</a>` : "N/A"}
      </div>
      <div class="schedule-field"><strong>Deadline:</strong> ${item.deadline}</div>
    `;
    container.appendChild(card);
  });
});
