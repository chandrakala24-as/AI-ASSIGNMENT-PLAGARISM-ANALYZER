// Admin Dashboard JavaScript Logic

let selectedFormRole = "student";

// Init Dashboard
document.addEventListener("DOMContentLoaded", () => {
    // Verify user role
    const currentUser = checkAuth(["admin"]);
    if (currentUser) {
        updateSidebarProfile(currentUser);
        loadUsersList();
    }
});

// Switch role between student and teacher in user form
function setFormRole(role) {
    selectedFormRole = role;
    
    document.getElementById("form-role-student").classList.remove("active");
    document.getElementById("form-role-teacher").classList.remove("active");
    
    document.getElementById(`form-role-${role}`).classList.add("active");
    
    const sectionGroup = document.getElementById("form-group-section");
    if (role === "student") {
        sectionGroup.classList.add("active");
    } else {
        sectionGroup.classList.remove("active");
    }
}

// Load users from backend
async function loadUsersList() {
    try {
        const users = await apiRequest(API_BASE + "/api/admin/users");
        
        const tableBody = document.getElementById("users-table-body");
        tableBody.innerHTML = "";
        
        let teacherCount = 0;
        let studentCount = 0;
        
        if (users.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-secondary);">No registered users found.</td></tr>`;
        } else {
            users.forEach(user => {
                if (user.role === "teacher") {
                    teacherCount++;
                } else if (user.role === "student") {
                    studentCount++;
                }
                
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td>
                        <span class="badge ${user.role === 'teacher' ? 'badge-violet' : 'badge-cyan'}">
                            ${user.role}
                        </span>
                    </td>
                    <td style="font-weight: 600;">${user.full_name}</td>
                    <td style="font-family: monospace;">${user.username}</td>
                    <td>${user.section ? `<span class="badge badge-yellow">${user.section}</span>` : '<span style="color: var(--text-muted); font-size: 0.8rem;">N/A</span>'}</td>
                    <td style="text-align: right;">
                        <button class="btn btn-danger btn-icon" onclick="deleteUser(${user.id}, '${user.username}')" title="Delete Account">
                            <i class="fa-solid fa-trash-can"></i>
                        </button>
                    </td>
                `;
                tableBody.appendChild(tr);
            });
        }
        
        // Update metric values
        document.getElementById("count-teachers").textContent = teacherCount;
        document.getElementById("count-students").textContent = studentCount;
        
    } catch (error) {
        console.error("Failed to load users:", error);
    }
}

// Create new user credentials
async function handleCreateUser(event) {
    event.preventDefault();
    
    const fullname = document.getElementById("new-fullname").value.trim();
    const username = document.getElementById("new-username").value.trim();
    const password = document.getElementById("new-password").value;
    const section = document.getElementById("new-section").value;
    const formAlert = document.getElementById("form-alert");
    
    formAlert.style.display = "none";
    
    try {
        const formData = new FormData();
        formData.append("username", username);
        formData.append("password", password);
        formData.append("full_name", fullname);
        formData.append("role", selectedFormRole);
        
        if (selectedFormRole === "student") {
            formData.append("section", section);
        }
        
        const response = await fetch(API_BASE + "/api/admin/users", {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Failed to create user account");
        }
        
        // Show success alert
        formAlert.className = "auth-alert auth-alert-success";
        formAlert.textContent = "Account credentials created successfully.";
        formAlert.style.display = "block";
        
        // Reset fields
        document.getElementById("new-fullname").value = "";
        document.getElementById("new-username").value = "";
        document.getElementById("new-password").value = "";
        
        // Reload list
        loadUsersList();
        
        // Hide success alert after 3 seconds
        setTimeout(() => {
            formAlert.style.display = "none";
        }, 3000);
        
    } catch (error) {
        formAlert.className = "auth-alert auth-alert-error";
        formAlert.textContent = error.message;
        formAlert.style.display = "block";
    }
}

// Delete user credentials
async function deleteUser(userId, username) {
    if (confirm(`Are you sure you want to permanently delete credentials for "${username}"?`)) {
        try {
            const response = await fetch(`${API_BASE}/api/admin/users/${userId}`, {
                method: "DELETE"
            });
            
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Failed to delete user");
            }
            
            // Reload list
            loadUsersList();
        } catch (error) {
            alert("Error deleting user: " + error.message);
        }
    }
}
