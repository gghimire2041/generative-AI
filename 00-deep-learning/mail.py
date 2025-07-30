"""
Incident Classification System using Pre-trained Embeddings + PyTorch with Class Descriptions
===========================================================================================

This implementation uses a separate embedding model to create text embeddings,
then feeds those embeddings along with class description context into PyTorch neural networks.

Approach:
1. Define classes with detailed descriptions
2. Generate fake incident data
3. Use pre-trained embedding model to create embeddings for both text and class descriptions
4. Train PyTorch neural networks using text embeddings and class description context
5. Provides multiple NN architectures with class-aware features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# =====================================================
# 1. CLASS DEFINITIONS WITH DESCRIPTIONS
# =====================================================

class IncidentClassDefinitions:
    def __init__(self):
        # Define the 7 incident classes with detailed descriptions
        self.class_definitions = {
            "Network Outage": {
                "description": "Network connectivity issues including internet outages, VPN failures, router malfunctions, switch problems, firewall blocking, DNS resolution issues, bandwidth limitations, and any infrastructure problems that prevent network communication and data transmission between systems, users, or external services.",
                "keywords": ["network", "connectivity", "internet", "VPN", "router", "switch", "firewall", "DNS", "bandwidth", "infrastructure", "communication", "transmission"],
                "severity_indicators": ["critical", "high priority", "service unavailable", "complete outage", "widespread impact"]
            },
            "Security Breach": {
                "description": "Security incidents involving unauthorized access attempts, malware infections, phishing attacks, data breaches, suspicious activities, brute force attacks, social engineering, vulnerability exploits, and any compromise of system integrity, confidentiality, or availability that threatens organizational security.",
                "keywords": ["security", "unauthorized", "malware", "phishing", "breach", "suspicious", "attack", "vulnerability", "compromise", "threat", "exploit", "intrusion"],
                "severity_indicators": ["security alert", "critical vulnerability", "data at risk", "immediate response", "containment required"]
            },
            "Hardware Failure": {
                "description": "Physical hardware malfunctions including server failures, disk crashes, CPU overheating, memory errors, power supply issues, cooling system problems, component degradation, and any physical equipment problems that affect system operation and require hardware replacement or repair.",
                "keywords": ["hardware", "server", "disk", "CPU", "memory", "power", "cooling", "component", "physical", "equipment", "replacement", "repair"],
                "severity_indicators": ["hardware failure", "system down", "replacement needed", "physical damage", "equipment malfunction"]
            },
            "Software Bug": {
                "description": "Software defects and programming errors including application crashes, unexpected behavior, logic errors, memory leaks, null pointer exceptions, API failures, database query issues, user interface problems, and any software-related issues that cause applications to malfunction or behave incorrectly.",
                "keywords": ["software", "bug", "crash", "error", "exception", "API", "database", "query", "application", "code", "programming", "logic"],
                "severity_indicators": ["application error", "system crash", "data corruption", "functionality broken", "critical bug"]
            },
            "User Access Issue": {
                "description": "Authentication and authorization problems including login failures, password resets, permission denied errors, account lockouts, single sign-on issues, two-factor authentication problems, user provisioning issues, and any problems preventing users from accessing systems or resources they need.",
                "keywords": ["access", "login", "password", "permission", "account", "authentication", "authorization", "SSO", "2FA", "user", "provisioning", "credentials"],
                "severity_indicators": ["access denied", "login failed", "account locked", "permission error", "authentication failure"]
            },
            "Performance Issue": {
                "description": "System performance degradation including slow response times, high resource utilization, memory consumption problems, CPU bottlenecks, database slowdowns, network latency, application timeouts, and any issues that cause systems to operate below expected performance levels or user experience standards.",
                "keywords": ["performance", "slow", "response", "latency", "bottleneck", "timeout", "resource", "utilization", "memory", "CPU", "degradation", "speed"],
                "severity_indicators": ["slow performance", "high latency", "resource exhaustion", "timeout errors", "degraded service"]
            },
            "Data Loss": {
                "description": "Data availability and integrity issues including file deletions, database corruption, backup failures, storage problems, data synchronization errors, recovery issues, and any incidents that result in loss, corruption, or unavailability of important business data or user information.",
                "keywords": ["data", "loss", "corruption", "backup", "recovery", "storage", "file", "database", "synchronization", "integrity", "deletion", "missing"],
                "severity_indicators": ["data lost", "corruption detected", "backup failed", "recovery needed", "critical data affected"]
            }
        }
        
        self.classes = list(self.class_definitions.keys())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
    
    def get_class_description(self, class_name):
        return self.class_definitions[class_name]["description"]
    
    def get_class_keywords(self, class_name):
        return self.class_definitions[class_name]["keywords"]
    
    def get_all_descriptions(self):
        return [self.class_definitions[cls]["description"] for cls in self.classes]

# =====================================================
# 2. FAKE DATA GENERATION WITH CLASS-AWARE CONTENT
# =====================================================

class IncidentDataGenerator:
    def __init__(self, class_definitions):
        self.class_definitions = class_definitions
        self.classes = class_definitions.classes
        
        # Enhanced templates that incorporate class-specific keywords
        self.templates = {
            "Network Outage": [
                "Network connectivity lost in {location}. Users unable to access {service}. {severity} impact on infrastructure communication.",
                "Internet connection down affecting {department}. Router malfunction causing {details} reported by users.",
                "VPN connectivity issues preventing remote access. DNS resolution failures and {investigation} in progress.",
                "Switch failure causing network instability in {location}. Firewall blocking legitimate traffic and {resolution} being implemented.",
                "Bandwidth limitations affecting {service}. Network infrastructure showing {details} requiring immediate attention.",
                "WiFi connectivity problems in {location}. Network transmission errors and communication breakdowns observed."
            ],
            "Security Breach": [
                "Suspicious login attempts detected from {location}. Unauthorized access threats and {action} taken immediately.",
                "Malware infection discovered on {system}. Security vulnerability exploited and {containment} measures activated.",
                "Phishing attack targeting {department}. Social engineering attempt with {response} protocol initiated.",
                "Brute force attack on {resource}. Multiple unauthorized access attempts blocked and security team investigating.",
                "Data breach attempt detected. Suspicious activities monitored and {security} containment procedures deployed.",
                "Intrusion detection system triggered. Compromise of system integrity suspected and immediate {action} required."
            ],
            "Hardware Failure": [
                "Server {server_id} experiencing critical hardware malfunction. Physical equipment failure causing {impact} on services.",
                "Disk crash detected on {system}. Storage hardware degradation and {backup} procedures initiated immediately.",
                "CPU overheating on {hardware}. Cooling system failure and {replacement} of physical components scheduled.",
                "Memory error causing system instability. Hardware component failure and {maintenance} repair procedures needed.",
                "Power supply unit failure on {server}. Equipment malfunction affecting {system} and redundancy systems activated.",
                "Hard drive showing bad sectors. Physical storage problems and {recovery} of hardware components required."
            ],
            "Software Bug": [
                "Application {app_name} crashing unexpectedly. Programming error and {error} detected in system logs.",
                "Database query optimization needed. Logic error causing {performance} degradation and code review required.",
                "API endpoint returning {error_code}. Software defect and {debugging} of application code in progress.",
                "User interface not responding correctly. Programming bug and {bug_report} submitted for code analysis.",
                "Memory leak detected in {application}. Software malfunction and {investigation} of code logic underway.",
                "Null pointer exception in {module}. Application error and {fix} being developed by programming team."
            ],
            "User Access Issue": [
                "User {user_id} unable to login to {system}. Authentication failure and {support} ticket created for access resolution.",
                "Password reset request for {user}. Credential problems and {verification} process initiated for account access.",
                "Permission denied error for {resource}. Authorization issue and {access} rights being reviewed by admin.",
                "Account locked due to multiple failed login attempts. Authentication security and {unlock} procedure started.",
                "Two-factor authentication failing for {user}. SSO problems and {troubleshooting} of user credentials in progress.",
                "Single sign-on not working for {application}. User provisioning issues and {configuration} being checked."
            ],
            "Performance Issue": [
                "System {system_name} running slowly. Performance degradation and {monitoring} shows high CPU resource utilization.",
                "Database queries taking longer than expected. Response time issues and {optimization} needed for better performance.",
                "Application timeout errors increasing. Performance bottleneck and {investigation} to identify slow components.",
                "Website loading slowly for users. Latency problems and {performance} analysis in progress for speed improvement.",
                "Batch job consuming excessive resources. High memory utilization and {resources} being allocated for performance.",
                "Server response time degraded significantly. Performance monitoring showing {analysis} of resource consumption needed."
            ],
            "Data Loss": [
                "Files missing from {directory}. Data deletion detected and {recovery} procedures initiated for lost information.",
                "Database corruption discovered during integrity check. Data loss event and {backup} restoration in progress.",
                "Email data accidentally deleted by user. File recovery needed and {retrieval} from backup systems started.",
                "User reports lost documents from {share}. Data synchronization failure and {investigation} of missing files.",
                "Backup verification failed completely. Storage corruption and {integrity} check revealing missing critical data.",
                "File system corruption on {storage}. Data availability issues and {recovery} tools being deployed immediately."
            ]
        }
        
        # Enhanced placeholders
        self.placeholders = {
            'location': ['Building A', 'Data Center', 'Remote Office', 'Main Campus', 'Branch Office'],
            'service': ['email system', 'web application', 'database', 'file server', 'CRM system'],
            'severity': ['Critical', 'High', 'Medium', 'Low'],
            'department': ['IT', 'Finance', 'HR', 'Sales', 'Marketing'],
            'details': ['Multiple complaints', 'Intermittent issues', 'Complete failure', 'Partial outage'],
            'investigation': ['Root cause analysis', 'Troubleshooting', 'Investigation', 'Diagnosis'],
            'resolution': ['Immediate fix', 'Temporary workaround', 'Scheduled maintenance', 'Emergency repair'],
            'action': ['Security lockdown', 'Immediate response', 'Investigation', 'Containment'],
            'system': ['production server', 'development environment', 'workstation', 'laptop'],
            'containment': ['Isolation', 'Quarantine', 'Removal', 'Blocking'],
            'resource': ['customer database', 'financial records', 'HR system', 'email server'],
            'user': ['John Smith', 'IT team', 'Security team', 'End user'],
            'response': ['Security', 'Emergency', 'Standard', 'Escalated'],
            'server_id': ['SRV-001', 'SRV-DB-01', 'SRV-WEB-02', 'SRV-APP-03'],
            'impact': ['Service unavailable', 'Degraded performance', 'Complete outage', 'Partial failure'],
            'backup': ['Emergency backup', 'Scheduled backup', 'Full restore', 'Incremental restore'],
            'hardware': ['main server', 'backup server', 'workstation cluster', 'storage array'],
            'replacement': ['Immediate', 'Next business day', 'Emergency', 'Planned'],
            'maintenance': ['Scheduled maintenance', 'Emergency repair', 'Component replacement', 'System upgrade'],
            'app_name': ['CRM Application', 'ERP System', 'Email Client', 'Web Portal'],
            'error': ['Stack overflow', 'Memory corruption', 'Null reference', 'Access violation'],
            'performance': ['Significant', 'Minor', 'Noticeable', 'Severe'],
            'error_code': ['500 error', '404 error', '403 error', 'timeout error'],
            'debugging': ['Code review', 'Log analysis', 'Performance profiling', 'Error tracing'],
            'bug_report': ['Detailed report', 'User feedback', 'Error log', 'Test case'],
            'application': ['web browser', 'desktop app', 'mobile app', 'service'],
            'fix': ['Hotfix', 'Patch', 'Update', 'Workaround'],
            'module': ['authentication module', 'payment processor', 'data handler', 'UI component'],
            'user_id': ['user123', 'jsmith', 'admin', 'testuser'],
            'support': ['Help desk', 'IT support', 'Technical', 'Emergency'],
            'verification': ['Identity', 'Email', 'Phone', 'Security'],
            'access': ['Read/write', 'Administrator', 'User', 'Guest'],
            'unlock': ['Manual', 'Automated', 'Security', 'Administrative'],
            'troubleshooting': ['Technical support', 'System diagnosis', 'Configuration check', 'Reset procedure'],
            'configuration': ['System settings', 'Network config', 'Security settings', 'User preferences'],
            'system_name': ['Production DB', 'File Server', 'Web Server', 'Mail Server'],
            'monitoring': ['System logs', 'Performance metrics', 'Resource usage', 'Health checks'],
            'optimization': ['Query tuning', 'Index optimization', 'Cache improvement', 'Resource allocation'],
            'resources': ['Additional CPU', 'More memory', 'Extra storage', 'Network bandwidth'],
            'analysis': ['Performance review', 'Resource audit', 'System check', 'Capacity planning'],
            'server': ['web server', 'database server', 'application server', 'file server'],
            'directory': ['/home/users', '/data/shared', '/backup/files', '/project/docs'],
            'recovery': ['Data recovery', 'File restoration', 'System rebuild', 'Backup restore'],
            'retrieval': ['Data recovery', 'File extraction', 'Backup restoration', 'Archive access'],
            'share': ['network drive', 'shared folder', 'cloud storage', 'team directory'],
            'integrity': ['Data validation', 'Checksum verification', 'Consistency check', 'Corruption scan'],
            'storage': ['SAN storage', 'NAS device', 'local disk', 'cloud storage'],
            'security': ['Security team', 'Incident response', 'Threat analysis', 'Vulnerability assessment']
        }
        
        self.ticket_patterns = ['INC-{:06d}', 'TICKET-{:05d}', 'REQ-{:06d}', 'ISSUE-{:05d}']
        
    def generate_text(self, incident_type, template):
        """Generate text by filling template with random placeholders"""
        text = template
        import re
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        for placeholder in placeholders:
            if placeholder in self.placeholders:
                value = random.choice(self.placeholders[placeholder])
                text = text.replace(f'{{{placeholder}}}', value)
        
        return text
    
    def generate_class_aware_content(self, incident_class):
        """Generate content that incorporates class-specific keywords"""
        class_keywords = self.class_definitions.get_class_keywords(incident_class)
        
        # Randomly include some class keywords in the content
        extra_keywords = random.sample(class_keywords, min(3, len(class_keywords)))
        keyword_text = f"Related to: {', '.join(extra_keywords)}."
        
        return keyword_text
    
    def generate_fake_data(self, n_samples=2000):
        """Generate fake incident data with class-aware content"""
        data = []
        
        for i in range(n_samples):
            # Randomly select incident class
            incident_class = random.choice(self.classes)
            class_idx = self.class_to_idx[incident_class]
            
            # Generate ticket ID
            ticket_id = random.choice(self.ticket_patterns).format(random.randint(1, 999999))
            
            # Generate ticket description
            template = random.choice(self.templates[incident_class])
            ticket_description = self.generate_text(incident_class, template)
            
            # Generate class-aware additional content
            class_context = self.generate_class_aware_content(incident_class)
            
            # Generate notes
            notes_templates = [
                f"User reported issue at {{time}}. Initial investigation shows {{finding}}. {class_context}",
                f"Escalated to {{team}} team. Priority set to {{priority}}. {class_context}",
                f"Workaround provided: {{workaround}}. Permanent fix scheduled. {class_context}",
                f"Root cause identified as {{cause}}. Resolution time estimated at {{time}}. {class_context}",
                f"Customer impact: {{impact}}. Business priority: {{priority}}. {class_context}",
                f"Technical details: {{technical}}. Next steps: {{steps}}. {class_context}"
            ]
            notes_template = random.choice(notes_templates)
            notes = self.generate_text(incident_class, notes_template)
            
            # Generate close notes
            close_templates = [
                f"Issue resolved by {{resolution}}. Verified by {{verifier}}. {class_context}",
                f"Problem fixed through {{fix}}. User confirmed resolution. {class_context}",
                f"Root cause addressed. Monitoring for {{duration}} to ensure stability. {class_context}",
                f"Temporary fix applied. Permanent solution scheduled for {{schedule}}. {class_context}",
                f"Issue closed after successful {{action}}. No further action required. {class_context}",
                f"Resolution confirmed. Documentation updated with {{info}}. {class_context}"
            ]
            close_template = random.choice(close_templates)
            close_notes = self.generate_text(incident_class, close_template)
            
            # Combine all text fields
            combined_text = f"Ticket: {ticket_description} Notes: {notes} Close Notes: {close_notes}"
            
            data.append({
                'ticket': ticket_id,
                'ticket_description': ticket_description,
                'notes': notes,
                'close_notes': close_notes,
                'combined': combined_text,
                'class': incident_class,
                'class_idx': class_idx
            })
        
        return pd.DataFrame(data)
    
    @property
    def class_to_idx(self):
        return self.class_definitions.class_to_idx

# =====================================================
# 3. ENHANCED EMBEDDING GENERATION WITH CLASS CONTEXT
# =====================================================

class ClassAwareEmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize embedding generator with sentence-transformers model
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
        
        self.class_embeddings = None
        self.class_descriptions = None
    
    def generate_class_embeddings(self, class_definitions):
        """Generate embeddings for class descriptions"""
        print("Generating class description embeddings...")
        descriptions = class_definitions.get_all_descriptions()
        self.class_descriptions = descriptions
        
        self.class_embeddings = self.model.encode(
            descriptions,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"Class embeddings shape: {self.class_embeddings.shape}")
        return self.class_embeddings
    
    def generate_text_embeddings(self, texts, batch_size=32):
        """Generate embeddings for incident texts"""
        print(f"Generating text embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def compute_class_similarities(self, text_embeddings):
        """Compute similarity scores between text and class descriptions"""
        if self.class_embeddings is None:
            raise ValueError("Class embeddings not generated. Call generate_class_embeddings first.")
        
        print("Computing class similarity features...")
        similarities = cosine_similarity(text_embeddings, self.class_embeddings)
        return similarities
    
    def create_enhanced_features(self, text_embeddings, include_similarities=True):
        """Create enhanced features combining text embeddings and class similarities"""
        features = [text_embeddings]
        
        if include_similarities and self.class_embeddings is not None:
            similarities = self.compute_class_similarities(text_embeddings)
            features.append(similarities)
            
            # Add normalized similarities
            normalized_similarities = similarities / np.linalg.norm(similarities, axis=1, keepdims=True)
            features.append(normalized_similarities)
            
            # Add max similarity scores
            max_similarities = np.max(similarities, axis=1, keepdims=True)
            features.append(max_similarities)
        
        enhanced_features = np.concatenate(features, axis=1)
        print(f"Enhanced features shape: {enhanced_features.shape}")
        return enhanced_features

# =====================================================
# 4. ENHANCED PYTORCH DATASET
# =====================================================

class EnhancedEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, class_similarities=None):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
        
        if class_similarities is not None:
            self.class_similarities = torch.FloatTensor(class_similarities)
        else:
            self.class_similarities = None
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        if self.class_similarities is not None:
            return self.embeddings[idx], self.labels[idx], self.class_similarities[idx]
        else:
            return self.embeddings[idx], self.labels[idx]

# =====================================================
# 5. CLASS-AWARE NEURAL NETWORK MODELS
# =====================================================

class ClassAwareMLPClassifier(nn.Module):
    """MLP with class description awareness"""
    def __init__(self, input_dim, num_classes, class_embedding_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super(ClassAwareMLPClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.class_embedding_dim = class_embedding_dim
        
        # Main feature processing
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Class-aware attention mechanism
        self.class_attention = nn.MultiheadAttention(
            embed_dim=prev_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(prev_dim, num_classes)
        )
    
    def forward(self, x, class_similarities=None):
        # Extract features
        features = self.feature_extractor(x)
        
        # Add batch and sequence dimensions for attention
        features_expanded = features.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Self-attention
        attended_features, _ = self.class_attention(
            features_expanded, features_expanded, features_expanded
        )
        
        # Remove sequence dimension
        attended_features = attended_features.squeeze(1)
        
        # If class similarities are provided, incorporate them
        if class_similarities is not None:
            # Weight features by class similarities
            similarity_weights = F.softmax(class_similarities, dim=1)
            similarity_weights = similarity_weights.unsqueeze(1)  # [batch, 1, num_classes]
            
            # Weighted combination (simplified approach)
            attended_features = attended_features + 0.1 * torch.sum(
                similarity_weights * class_similarities.unsqueeze(1), dim=2
            )
        
        # Final classification
        output = self.classifier(attended_features)
        return output

class ClassAwareLSTMClassifier(nn.Module):
    """LSTM with class description awareness"""
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super(ClassAwareLSTMClassifier, self).__init__()
        
        # Input processing
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # Class similarity integration
        self.class_integration = nn.Linear(num_classes, hidden_dim * 2)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, class_similarities=None):
        batch_size = x.size(0)
        
        # Create sequence from embeddings
        seq_len = max(8, x.size(1) // 8)
        if x.size(1) % seq_len != 0:
            padding_size = seq_len - (x.size(1) % seq_len)
            x = F.pad(x, (0, padding_size))
        
        x = x[:, :seq_len * 8].view(batch_size, seq_len, 8)
        
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Integrate class similarities if provided
        if class_similarities is not None:
            class_features = self.class_integration(class_similarities)
            pooled = pooled + 0.2 * class_features
        
        # Classification
        output = self.classifier(pooled)
        return output

class ClassAwareCNNClassifier(nn.Module):
    """CNN with class description awareness"""
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(ClassAwareCNNClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.feature_size = int((input_dim - num_classes) ** 0.5)
        
        # Adjust input dimension
        actual_input_dim = input_dim - num_classes  # Separate class similarities
        if self.feature_size ** 2 != actual_input_dim:
            self.input_projection = nn.Linear(actual_input_dim, self.feature_size ** 2)
        else:
            self.input_projection = None
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Class similarity integration
        self.class_integration = nn.Linear(num_classes, 128)
        
        # Classifier
        conv_output_size = 128 * 4 * 4
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(conv_output_size + 128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, class_similarities=None):
        batch_size = x.size(0)
        
        # Separate embeddings and class similarities if concatenated
        if class_similarities is None and x.size(1) > self.feature_size ** 2:
            embeddings = x[:, :-self.num_classes]
            class_similarities = x[:, -self.num_classes:]
        else:
            embeddings = x
        
        # Project embeddings if needed
        if self.input_projection:
            embeddings = self.input_projection(embeddings)
            dim = self.feature_size
        else:
            dim = self.feature_size
        
        # Reshape to 2D
        embeddings = embeddings.view(batch_size, 1, dim, dim)
        
        # Apply convolutions
        conv_features = self.conv_layers(embeddings)
        conv_features = conv_features.view(batch_size, -1)
        
        # Integrate class similarities
        if class_similarities is not None:
            class_features = self.class_integration(class_similarities)
            combined_features = torch.cat([conv_features, class_features], dim=1)
        else:
            # Add zero features if no class similarities
            zero_class_features = torch.zeros(batch_size, 128, device=conv_features.device)
            combined_features = torch.cat([conv_features, zero_class_features], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        return output

# =====================================================
# 6. ENHANCED TRAINING AND EVALUATION
# =====================================================

class ClassAwareModelTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_accuracies = []
    
    def train_model(self, train_loader, val_loader, num_epochs=50, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        best_val_acc = 0.0
        best_model_state = None
        
        print(f"Training on {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
                for batch in pbar:
                    if len(batch) == 3:  # With class similarities
                        embeddings, labels, class_similarities = batch
                        embeddings = embeddings.to(self.device)
                        labels = labels.to(self.device) 
                        class_similarities = class_similarities.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = self.model(embeddings, class_similarities)
                    else:  # Without class similarities
                        embeddings, labels = batch
                        embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = self.model(embeddings)
                    
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100 * train_correct / train_total:.2f}%'
                    })
            
            # Validation phase
            val_acc = self.evaluate_model(val_loader)
            
            self.train_losses.append(train_loss / len(train_loader))
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Acc: {100 * train_correct / train_total:.2f}%, '
                  f'Val Acc: {val_acc:.2f}%, '
                  f'Best Val Acc: {best_val_acc:.2f}%')
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return best_val_acc
    
    def evaluate_model(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:  # With class similarities
                    embeddings, labels, class_similarities = batch
                    embeddings = embeddings.to(self.device)
                    labels = labels.to(self.device)
                    class_similarities = class_similarities.to(self.device)
                    outputs = self.model(embeddings, class_similarities)
                else:  # Without class similarities
                    embeddings, labels = batch
                    embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                    outputs = self.model(embeddings)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def get_predictions(self, data_loader):
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:  # With class similarities
                    embeddings, labels, class_similarities = batch
                    embeddings = embeddings.to(self.device)
                    labels = labels.to(self.device)
                    class_similarities = class_similarities.to(self.device)
                    outputs = self.model(embeddings, class_similarities)
                else:  # Without class similarities
                    embeddings, labels = batch
                    embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                    outputs = self.model(embeddings)
                
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

# =====================================================
# 7. VISUALIZATION AND ANALYSIS
# =====================================================

def plot_class_similarities(text_embeddings, class_embeddings, class_names, sample_size=100):
    """Plot similarity heatmap between sample texts and class descriptions"""
    # Take a random sample for visualization
    sample_indices = np.random.choice(len(text_embeddings), min(sample_size, len(text_embeddings)), replace=False)
    sample_embeddings = text_embeddings[sample_indices]
    
    # Compute similarities
    similarities = cosine_similarity(sample_embeddings, class_embeddings)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(similarities, 
                xticklabels=class_names,
                yticklabels=[f'Text {i}' for i in range(len(sample_embeddings))],
                cmap='YlOrRd', 
                center=0)
    plt.title('Text-to-Class Description Similarities')
    plt.xlabel('Class Descriptions')
    plt.ylabel('Sample Texts')
    plt.tight_layout()
    plt.show()

def analyze_class_awareness(predictions, true_labels, class_similarities, class_names):
    """Analyze how class similarities correlate with predictions"""
    correct_mask = predictions == true_labels
    
    print("\nClass Awareness Analysis:")
    print("=" * 50)
    
    for i, class_name in enumerate(class_names):
        class_mask = true_labels == i
        if np.sum(class_mask) == 0:
            continue
            
        # Average similarity for this class
        avg_similarity = np.mean(class_similarities[class_mask, i])
        
        # Accuracy for this class
        class_accuracy = np.mean(correct_mask[class_mask])
        
        print(f"{class_name}:")
        print(f"  Average similarity to description: {avg_similarity:.4f}")
        print(f"  Classification accuracy: {class_accuracy:.4f}")
        print()

def plot_training_history(trainer, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(trainer.train_losses)
    ax1.set_title(f'{model_name} - Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot validation accuracy
    ax2.plot(trainer.val_accuracies)
    ax2.set_title(f'{model_name} - Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def analyze_predictions(y_true, y_pred, y_prob, class_names, model_name):
    print(f"\n{model_name} - Detailed Results:")
    print("=" * 50)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

# =====================================================
# 8. MAIN EXECUTION FUNCTION
# =====================================================

def main():
    print("Class-Aware Incident Classification with Embeddings + PyTorch")
    print("=" * 70)
    
    # 1. Initialize class definitions
    print("\n1. Initializing class definitions...")
    class_definitions = IncidentClassDefinitions()
    
    print("Class Definitions:")
    for i, (class_name, definition) in enumerate(class_definitions.class_definitions.items()):
        print(f"\n{i+1}. {class_name}:")
        print(f"   Description: {definition['description'][:100]}...")
        print(f"   Keywords: {', '.join(definition['keywords'][:5])}...")
    
    # 2. Generate fake data
    print(f"\n2. Generating fake incident data...")
    generator = IncidentDataGenerator(class_definitions)
    df = generator.generate_fake_data(n_samples=2000)
    
    print(f"Generated {len(df)} samples across {len(class_definitions.classes)} classes")
    print("\nClass distribution:")
    print(df['class'].value_counts().sort_index())
    
    # Display examples
    print("\nExample incidents:")
    for i, class_name in enumerate(class_definitions.classes[:3]):
        example = df[df['class'] == class_name].iloc[0]
        print(f"\n{class_name}:")
        print(f"Combined text: {example['combined'][:150]}...")
    
    # 3. Generate embeddings with class awareness
    print("\n3. Generating class-aware embeddings...")
    embedding_generator = ClassAwareEmbeddingGenerator('all-MiniLM-L6-v2')
    
    # Generate class description embeddings
    class_embeddings = embedding_generator.generate_class_embeddings(class_definitions)
    
    # Generate text embeddings
    text_embeddings = embedding_generator.generate_text_embeddings(df['combined'].tolist())
    
    # Create enhanced features with class similarities
    enhanced_features = embedding_generator.create_enhanced_features(text_embeddings)
    
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Class embeddings shape: {class_embeddings.shape}")
    print(f"Enhanced features shape: {enhanced_features.shape}")
    
    # 4. Prepare data for training
    print("\n4. Preparing data for training...")
    
    # Get class similarities for analysis
    class_similarities = embedding_generator.compute_class_similarities(text_embeddings)
    
    X_train, X_test, y_train, y_test, sim_train, sim_test = train_test_split(
        enhanced_features, df['class_idx'].values, class_similarities,
        test_size=0.2, 
        random_state=42, 
        stratify=df['class_idx'].values
    )
    
    X_train, X_val, y_train, y_val, sim_train, sim_val = train_test_split(
        X_train, y_train, sim_train,
        test_size=0.2, 
        random_state=42, 
        stratify=y_train
    )
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Create datasets with class similarities
    train_dataset = EnhancedEmbeddingDataset(X_train, y_train, sim_train)
    val_dataset = EnhancedEmbeddingDataset(X_val, y_val, sim_val)
    test_dataset = EnhancedEmbeddingDataset(X_test, y_test, sim_test)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 5. Train class-aware models
    print("\n5. Training class-aware models...")
    
    input_dim = enhanced_features.shape[1]
    num_classes = len(class_definitions.classes)
    class_embedding_dim = class_embeddings.shape[1]
    
    models_to_train = {
        'ClassAware-MLP': ClassAwareMLPClassifier(input_dim, num_classes, class_embedding_dim),
        'ClassAware-LSTM': ClassAwareLSTMClassifier(input_dim, 128, 2, num_classes),
        'ClassAware-CNN': ClassAwareCNNClassifier(input_dim, num_classes)
    }
    
    results = {}
    
    for model_name, model in models_to_train.items():
        print(f"\nTraining {model_name}...")
        trainer = ClassAwareModelTrainer(model)
        
        # Train model
        best_val_acc = trainer.train_model(
            train_loader, val_loader, 
            num_epochs=30, lr=0.001
        )
        
        # Evaluate on test set
        test_acc = trainer.evaluate_model(test_loader)
        y_pred, y_true, y_prob = trainer.get_predictions(test_loader)
        
        results[model_name] = {
            'trainer': trainer,
            'test_accuracy': test_acc,
            'predictions': (y_pred, y_true, y_prob)
        }
        
        print(f"{model_name} Test Accuracy: {test_acc:.2f}%")
    
    # 6. Compare results and analyze class awareness
    print("\n6. Model Comparison:")
    print("=" * 40)
    for model_name, result in results.items():
        print(f"{model_name}: {result['test_accuracy']:.2f}%")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    print(f"\nBest Model: {best_model_name}")
    
    # 7. Detailed analysis
    print(f"\n7. Detailed analysis of {best_model_name}:")
    best_result = results[best_model_name]
    y_pred, y_true, y_prob = best_result['predictions']
    
    analyze_predictions(y_true, y_pred, y_prob, class_definitions.classes, best_model_name)
    analyze_class_awareness(y_pred, y_true, sim_test, class_definitions.classes)
    
    # 8. Visualizations
    print("\n8. Generating visualizations...")
    
    # Plot class similarities
    plot_class_similarities(text_embeddings, class_embeddings, class_definitions.classes)
    
    # Plot training history for best model
    plot_training_history(best_result['trainer'], best_model_name)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_definitions.classes, best_model_name)
    
    # 9. Save the best model
    print(f"\n9. Saving best model ({best_model_name})...")
    torch.save({
        'model_state_dict': best_result['trainer'].model.state_dict(),
        'model_class': best_result['trainer'].model.__class__.__name__,
        'classes': class_definitions.classes,
        'class_definitions': class_definitions.class_definitions,
        'embedding_model': 'all-MiniLM-L6-v2',
        'input_dim': input_dim,
        'num_classes': num_classes,
        'class_embeddings': class_embeddings
    }, f'best_class_aware_classifier_{best_model_name.replace("-", "_").lower()}.pth')
    
    print("Training completed!")
    
    return results, class_definitions, embedding_generator

# Example usage for inference with class awareness
def class_aware_inference_example(model_path, embedding_generator, text, class_definitions):
    """Example of how to use the trained class-aware model for inference"""
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Generate text embedding
    text_embedding = embedding_generator.generate_text_embeddings([text])
    
    # Create enhanced features
    enhanced_features = embedding_generator.create_enhanced_features(text_embedding)
    
    # Get class similarities
    class_similarities = embedding_generator.compute_class_similarities(text_embedding)
    
    print(f"Text: {text}")
    print(f"Enhanced features shape: {enhanced_features.shape}")
    print("\nClass Similarities:")
    for i, class_name in enumerate(class_definitions.classes):
        similarity = class_similarities[0, i]
        print(f"  {class_name}: {similarity:.4f}")
    
    print("\nPredicted class would be determined using the saved model...")

if __name__ == "__main__":
    # Install required packages first:
    # pip install torch sentence-transformers pandas scikit-learn matplotlib seaborn tqdm
    
    results, class_definitions, embedding_gen = main()
