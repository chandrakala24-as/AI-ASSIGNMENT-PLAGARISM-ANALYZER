// Student Dashboard JavaScript Logic

let studentUser = null;
let selectedFile = null;
let activeSubmissions = [];
let submissionsPollInterval = null;

document.addEventListener("DOMContentLoaded", () => {
    studentUser = checkAuth(["student"]);
    if (studentUser) {
        updateSidebarProfile(studentUser);
        document.getElementById("welcome-message").textContent = `Welcome back, ${studentUser.full_name}`;
        document.getElementById("student-section-label").innerHTML = `<i class="fa-solid fa-graduation-cap" style="color: var(--accent-cyan);"></i> Class Section: <strong>${studentUser.section}</strong>`;
        
        loadStudentDashboard();
    }
    
    // Drag & drop setup
    const dropZone = document.getElementById("drop-zone");
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragging');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragging');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            setUploadedFile(files[0]);
        }
    });
});

async function loadStudentDashboard() {
    try {
        await Promise.all([
            loadAssignments(),
            loadSubmissions()
        ]);
    } catch (err) {
        console.error("Error loading dashboard data:", err);
    }
}

// Fetch and render assignments for section
async function loadAssignments() {
    const assignments = await apiRequest(`/api/student/assignments?section=${studentUser.section}`);
    
    // Select dropdown
    const select = document.getElementById("select-assignment");
    select.innerHTML = '<option value="" disabled selected>-- Choose Assignment Prompt --</option>';
    
    if (assignments.length === 0) {
        return;
    }
    
    assignments.forEach(asg => {
        // Dropdown option
        const opt = document.createElement("option");
        opt.value = asg.id;
        opt.textContent = asg.title;
        select.appendChild(opt);
    });
}

// Fetch and render submissions
async function loadSubmissions() {
    const subs = await apiRequest(`/api/student/submissions?student_id=${studentUser.user_id}`);
    activeSubmissions = subs;
    
    const tableBody = document.getElementById("submissions-table-body");
    tableBody.innerHTML = "";
    
    let pendingCount = 0;
    let gradedCount = 0;
    let plagSum = 0;
    let reportCount = 0;
    
    if (subs.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-secondary);">No assignments submitted yet.</td></tr>`;
    } else {
        subs.forEach(sub => {
            if (sub.marks !== null) {
                gradedCount++;
            }
            
            let plagText = '<span style="color: var(--text-muted);">Scanning...</span>';
            if (sub.overall_plagiarism_pct !== null) {
                const pct = sub.overall_plagiarism_pct;
                plagSum += pct;
                reportCount++;
                
                let badgeClass = "badge-green";
                if (pct >= 50) badgeClass = "badge-red";
                else if (pct >= 20) badgeClass = "badge-yellow";
                
                plagText = `<span class="badge ${badgeClass}">${pct}% Copy</span>`;
            } else {
                pendingCount++;
            }
            
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td style="font-weight: 600; color: var(--text-primary);">${sub.assignment_title}</td>
                <td style="font-family: monospace; font-size: 0.8rem; max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${sub.file_name}</td>
                <td>${formatDate(sub.submitted_at)}</td>
                <td>${plagText}</td>
                <td style="font-weight: 700;">
                    ${sub.marks !== null ? `<span style="color: var(--accent-green);">${sub.marks} / 100</span>` : '<span style="color: var(--text-muted); font-size: 0.85rem;">Ungraded</span>'}
                </td>
                <td style="text-align: right;">
                    ${sub.overall_plagiarism_pct !== null ? `
                        <button class="btn btn-primary" style="padding: 6px 12px; font-size: 0.8rem;" onclick="viewReport('${sub.submission_id}')">
                            <i class="fa-solid fa-square-poll-vertical"></i> View Report
                        </button>
                    ` : `
                        <button class="btn btn-secondary" style="padding: 6px 12px; font-size: 0.8rem;" disabled>
                            <i class="fa-solid fa-spinner fa-spin"></i> Processing
                        </button>
                    `}
                </td>
            `;
            tableBody.appendChild(tr);
        });
    }
    
    // Update metric dashboard values
    document.getElementById("count-pending-tasks").textContent = pendingCount;
    document.getElementById("count-graded-tasks").textContent = gradedCount;
    document.getElementById("count-avg-plag").textContent = reportCount > 0 ? `${Math.round(plagSum / reportCount)}%` : "0%";
    document.getElementById("count-total-subs").textContent = subs.length;

    // Automatic polling while submissions are processing
    if (pendingCount > 0) {
        if (!submissionsPollInterval) {
            submissionsPollInterval = setInterval(() => {
                loadSubmissions();
            }, 3000);
        }
    } else {
        if (submissionsPollInterval) {
            clearInterval(submissionsPollInterval);
            submissionsPollInterval = null;
        }
    }
}

// Drag & drop selection handler
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        setUploadedFile(files[0]);
    }
}

function setUploadedFile(file) {
    selectedFile = file;
    
    // UI Updates
    document.getElementById("selected-file-container").style.display = "block";
    document.getElementById("file-display-name").textContent = file.name;
    document.getElementById("file-display-size").textContent = formatBytes(file.size);
    
    // Change file type icon
    const fileIcon = document.getElementById("primary-upload-icon");
    const ext = file.name.split('.').pop().toLowerCase();
    fileIcon.className = "upload-icon";
    if (ext === 'pdf') {
        fileIcon.classList.add('fa-solid', 'fa-file-pdf');
        fileIcon.style.color = '#ff3366';
    } else if (ext === 'docx') {
        fileIcon.classList.add('fa-solid', 'fa-file-word');
        fileIcon.style.color = '#3b82f6';
    } else if (ext === 'pptx' || ext === 'ppt') {
        fileIcon.classList.add('fa-solid', 'fa-file-powerpoint');
        fileIcon.style.color = '#ff8c21';
    } else if (['png', 'jpg', 'jpeg'].includes(ext)) {
        fileIcon.classList.add('fa-solid', 'fa-file-image');
        fileIcon.style.color = '#22ff6e';
    } else {
        fileIcon.classList.add('fa-solid', 'fa-file-lines');
        fileIcon.style.color = '#a0a8c8';
    }
    
    document.getElementById("btn-upload").removeAttribute("disabled");
}

function resetFileSelection() {
    selectedFile = null;
    document.getElementById("file-input").value = "";
    document.getElementById("selected-file-container").style.display = "none";
    
    const fileIcon = document.getElementById("primary-upload-icon");
    fileIcon.className = "fa-solid fa-cloud-arrow-up upload-icon";
    fileIcon.style.color = "var(--text-secondary)";
    
    document.getElementById("btn-upload").setAttribute("disabled", "true");
}

// Handle Form Submission
async function handleUploadSubmit(event) {
    event.preventDefault();
    
    const assignmentId = document.getElementById("select-assignment").value;
    const alertBox = document.getElementById("upload-alert");
    
    if (!assignmentId) {
        alert("Please choose an assignment prompt from the list.");
        return;
    }
    
    if (!selectedFile) {
        alert("Please upload or drag a document file.");
        return;
    }
    
    alertBox.style.display = "none";
    
    const btnUpload = document.getElementById("btn-upload");
    const btnText = btnUpload.querySelector("span");
    const btnIcon = btnUpload.querySelector("i");
    
    btnUpload.setAttribute("disabled", "true");
    btnText.textContent = "Uploading & Analyzing...";
    btnIcon.className = "fa-solid fa-circle-notch fa-spin";
    
    try {
        const formData = new FormData();
        formData.append("assignment_id", assignmentId);
        formData.append("student_id", studentUser.user_id);
        formData.append("file", selectedFile);
        
        const response = await fetch("/api/student/upload", {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Upload failed");
        }
        
        const result = await response.json();
        
        alertBox.className = "auth-alert auth-alert-success";
        alertBox.textContent = "Assignment uploaded successfully! The AI engine is analyzing plagiarism ratios. Please stand by.";
        alertBox.style.display = "block";
        
        // Reset selections
        resetFileSelection();
        document.getElementById("select-assignment").value = "";
        
        // Reload table
        await loadStudentDashboard();
        
        // Hide notice after 4 seconds
        setTimeout(() => {
            alertBox.style.display = "none";
        }, 4000);
        
    } catch (error) {
        alertBox.className = "auth-alert auth-alert-error";
        alertBox.textContent = error.message;
        alertBox.style.display = "block";
    } finally {
        btnUpload.removeAttribute("disabled");
        btnText.textContent = "Upload & Submit Report";
        btnIcon.className = "fa-solid fa-paper-plane";
    }
}

// Render Plagiarism Highlights Report Modal
function viewReport(submissionId) {
    const sub = activeSubmissions.find(s => s.submission_id === submissionId);
    if (!sub || !sub.detailed_report) return;
    
    const rep = sub.detailed_report;
    const text = sub.extracted_text || "";
    
    // Set title and metrics
    document.getElementById("modal-report-title").textContent = `Plagiarism Report: ${sub.assignment_title}`;
    
    document.getElementById("modal-val-overall").textContent = `${Math.round(sub.overall_plagiarism_pct)}%`;
    document.getElementById("modal-progress-overall").style.background = `conic-gradient(var(--accent-red) ${sub.overall_plagiarism_pct * 3.6}deg, rgba(255, 255, 255, 0.05) 0deg)`;
    
    document.getElementById("modal-val-peer").textContent = `${Math.round(sub.peer_similarity_pct)}%`;
    document.getElementById("modal-progress-peer").style.background = `conic-gradient(var(--accent-violet) ${sub.peer_similarity_pct * 3.6}deg, rgba(255, 255, 255, 0.05) 0deg)`;
    
    document.getElementById("modal-val-internet").textContent = `${Math.round(sub.internet_similarity_pct)}%`;
    document.getElementById("modal-progress-internet").style.background = `conic-gradient(var(--accent-cyan) ${sub.internet_similarity_pct * 3.6}deg, rgba(255, 255, 255, 0.05) 0deg)`;
    
    // Set marks and feedback
    const marksText = document.getElementById("modal-text-marks");
    const feedbackText = document.getElementById("modal-text-feedback");
    
    if (sub.marks !== null) {
        marksText.innerHTML = `<span style="color: var(--accent-green); font-size: 1.1rem; font-weight: 700;">${sub.marks} / 100</span>`;
        feedbackText.textContent = sub.feedback || "No feedback text left by the teacher.";
    } else {
        marksText.textContent = "Pending Grade";
        feedbackText.textContent = "Teacher has not graded this report yet.";
    }
    
    // Generate highlighted text
    const docViewer = document.getElementById("modal-doc-viewer");
    docViewer.innerHTML = generateHighlightedText(text, rep.peer_matches, rep.internet_matches);
    
    // Open Modal
    document.getElementById("report-modal").classList.add("active");
}

function closeReportModal() {
    document.getElementById("report-modal").classList.remove("active");
}

// Generate colored span highlight elements
function generateHighlightedText(text, peerMatches, internetMatches) {
    if (!text) return '<span style="color: var(--text-muted); font-style: italic;">[No content extracted from document]</span>';
    
    // Create char highlight tag arrays
    // type: 0 = none, 1 = peer, 2 = internet
    const highlightType = new Array(text.length).fill(0);
    const highlightSource = new Array(text.length).fill("");
    
    // Populate Internet Matches (lower priority, overwritable by peer)
    if (internetMatches) {
        internetMatches.forEach(src => {
            if (src.matched_spans) {
                src.matched_spans.forEach(span => {
                    const start = span.start1;
                    const end = span.end1;
                    for (let i = start; i < end && i < highlightType.length; i++) {
                        highlightType[i] = 2;
                        highlightSource[i] = `${src.title} (${src.url})`;
                    }
                });
            }
        });
    }
    
    // Populate Peer Matches (higher priority)
    if (peerMatches) {
        peerMatches.forEach(peer => {
            if (peer.matched_spans) {
                peer.matched_spans.forEach(span => {
                    const start = span.start1;
                    const end = span.end1;
                    for (let i = start; i < end && i < highlightType.length; i++) {
                        highlightType[i] = 1;
                        highlightSource[i] = `${peer.student_name} (${peer.student_section})`;
                    }
                });
            }
        });
    }
    
    // Construct HTML string
    let html = [];
    let i = 0;
    
    while (i < text.length) {
        const type = highlightType[i];
        const src = highlightSource[i];
        
        if (type === 0) {
            // Find length of unhighlighted segment
            let start = i;
            while (i < text.length && highlightType[i] === 0) {
                i++;
            }
            // Escape HTML
            html.push(escapeHtml(text.substring(start, i)));
        } else {
            // Highlighted segment
            let start = i;
            while (i < text.length && highlightType[i] === type && highlightSource[i] === src) {
                i++;
            }
            
            const className = type === 1 ? "highlight-peer" : "highlight-internet";
            html.push(`<span class="${className}" data-source="${escapeHtml(src)}">${escapeHtml(text.substring(start, i))}</span>`);
        }
    }
    
    return html.join("");
}

function escapeHtml(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}
