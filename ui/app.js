let currentTab = 'PENDING';

document.addEventListener('DOMContentLoaded', () => {
    fetchProposals('PENDING');

    document.getElementById('train-lora-btn').addEventListener('click', triggerTraining);
    
    document.getElementById('tab-pending').addEventListener('click', (e) => switchTab(e, 'PENDING'));
    document.getElementById('tab-approved').addEventListener('click', (e) => switchTab(e, 'APPROVED'));
    document.getElementById('tab-metrics').addEventListener('click', (e) => switchTab(e, 'METRICS'));
});

function switchTab(e, tabName) {
    e.preventDefault();
    document.querySelectorAll('.menu a').forEach(a => a.classList.remove('active'));
    e.target.classList.add('active');
    
    currentTab = tabName;
    const headerTitle = document.querySelector('header h1');
    
    if (tabName === 'METRICS') {
        headerTitle.innerText = "Model Metrics";
        document.getElementById('pending-count').style.display = 'none';
        fetchMetrics();
    } else {
        headerTitle.innerText = tabName === 'PENDING' ? "Pending Proposals" : "Approved Pairs";
        document.getElementById('pending-count').style.display = 'inline-block';
        fetchProposals(tabName);
    }
}

async function fetchMetrics() {
    const grid = document.getElementById('proposals-grid');
    grid.innerHTML = '<p style="color: var(--text-muted);">Loading metrics...</p>';
    
    try {
        const res = await fetch('/api/metrics');
        const data = await res.json();
        
        grid.innerHTML = `
            <div class="card" style="border-color: var(--accent-glow);">
                <div class="card-header"><span class="badge safe">DPO Training</span></div>
                <div class="card-body">
                    <div class="label">Final DPO Loss</div>
                    <div class="text-content" style="font-size: 1.5rem; color: white;">${data.dpo_final_loss}</div>
                    <div class="label" style="margin-top: 1rem;">Chosen Accuracy</div>
                    <div class="text-content" style="font-size: 1.5rem; color: var(--success);">${data.dpo_chosen_acc}</div>
                </div>
            </div>
            <div class="card">
                <div class="card-header"><span class="badge unknown">Database Stats</span></div>
                <div class="card-body">
                    <div class="label">Total Approved Pairs</div>
                    <div class="text-content" style="font-size: 1.5rem; color: white;">${data.proposals_approved}</div>
                    <div class="label" style="margin-top: 1rem;">Total Rejected Pairs</div>
                    <div class="text-content" style="font-size: 1.5rem; color: var(--danger);">${data.proposals_rejected}</div>
                </div>
            </div>
        `;
    } catch (err) {
        grid.innerHTML = '<p style="color: var(--danger);">Failed to load metrics.</p>';
    }
}

async function fetchProposals(status) {
    try {
        const response = await fetch(`/api/proposals?status=${status}`);
        const proposals = await response.json();
        
        if (status === 'PENDING') {
            document.getElementById('pending-count').innerText = `${proposals.length} Pending`;
        } else {
            document.getElementById('pending-count').innerText = `${proposals.length} Approved`;
        }
        
        const grid = document.getElementById('proposals-grid');
        grid.innerHTML = ''; // Clear current
        
        if (proposals.length === 0) {
            grid.innerHTML = `<p style="color: var(--text-muted);">No ${status.toLowerCase()} proposals.</p>`;
            return;
        }

        proposals.forEach(p => {
            const card = document.createElement('div');
            card.className = 'card';
            
            let riskClass = 'unknown';
            if (p.risk_level === 'SAFE' || p.risk_level === 'LOW') riskClass = 'safe';
            if (p.risk_level === 'HIGH_RISK' || p.risk_level === 'HIGH') riskClass = 'risky';

            let actionsHtml = '';
            if (status === 'PENDING') {
                actionsHtml = `
                    <div class="card-actions">
                        <button class="action-btn approve" onclick="reviewProposal(${p.id}, 'APPROVED')">Approve</button>
                        <button class="action-btn reject" onclick="reviewProposal(${p.id}, 'REJECTED')">Reject</button>
                    </div>
                `;
            } else {
                actionsHtml = `
                    <div class="card-actions">
                        <span style="color: var(--text-muted); font-size: 0.85rem;">Status: ${p.status}</span>
                    </div>
                `;
            }

            card.innerHTML = `
                <div class="card-header">
                    <span class="id-tag" style="color: var(--text-muted); font-size: 0.8rem;">#${p.id}</span>
                    <span class="badge ${riskClass}">${p.risk_level}</span>
                </div>
                <div class="card-body">
                    <div>
                        <div class="label">Question</div>
                        <div class="text-content" style="font-weight: 500;">${p.question}</div>
                    </div>
                    <div>
                        <div class="label">Proposed Answer</div>
                        <div class="text-content" style="color: #cbd5e1;">${p.proposed_answer}</div>
                    </div>
                </div>
                ${actionsHtml}
            `;
            grid.appendChild(card);
        });
    } catch (err) {
        console.error("Failed to load proposals", err);
    }
}

async function reviewProposal(id, action) {
    try {
        const res = await fetch('/api/review', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ proposal_id: id, action: action })
        });
        if (res.ok) {
            fetchProposals(currentTab); // refresh
        }
    } catch (err) {
        console.error("Failed to submit review", err);
    }
}

async function triggerTraining() {
    const btn = document.getElementById('train-lora-btn');
    const status = document.getElementById('train-status');
    
    btn.disabled = true;
    btn.innerText = "Training...";
    
    try {
        const res = await fetch('/api/train_lora', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        
        const data = await res.json();
        
        if (res.ok) {
            status.style.color = "var(--success)";
            status.innerText = data.message;
        } else {
            status.style.color = "var(--danger)";
            status.innerText = data.detail || "Error starting training";
        }
    } catch (err) {
        status.style.color = "var(--danger)";
        status.innerText = "Connection error";
    } finally {
        setTimeout(() => {
            btn.disabled = false;
            btn.innerText = "Train LoRA Now";
        }, 3000);
    }
}
