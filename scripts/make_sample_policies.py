"""
Generates the sample policy PDFs in data/policies/ for a fictional company,
Example Manufacturing GmbH. They are the corpus for the demo and the evaluation.

    python scripts/make_sample_policies.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "policies"
COMPANY = "Example Manufacturing GmbH"

POLICIES = {
    "IS-01_Information_Security_Policy": {
        "title": "Information Security Policy",
        "meta": "Document IS-01, version 3.2, owner: Chief Information Security Officer (CISO), effective 1 March 2026",
        "sections": [
            ("Purpose and Scope", [
                "This policy sets the minimum security requirements for protecting company information and IT systems. "
                "It applies to all employees, contractors and third parties with access to company information systems "
                "at all sites in Germany and the Czech Republic.",
                "The information security management system is aligned with ISO/IEC 27001:2022. Topic-specific rules "
                "are defined in this policy and in the related procedures, such as the Incident Response Procedure (IS-07).",
            ]),
            ("Roles and Responsibilities", [
                "The CISO owns this policy and reviews it at least once a year. IT Operations implements the technical "
                "controls described here. Managers make sure that everyone in their team completes the annual security "
                "awareness training by 31 March of each year.",
                "Every employee must protect the information entrusted to them and report security incidents and "
                "weaknesses without delay.",
            ]),
            ("Access Control", [
                "Access is granted on the principle of least privilege. Access requests are submitted through the IT "
                "service portal and must be approved by the employee's manager and, for business applications, by the "
                "application owner.",
                "Access rights are reviewed every 6 months for standard systems. Privileged accounts and access to the "
                "SAP ERP system are reviewed every 3 months. Accounts of leavers are disabled on their last working day.",
                "Shared user accounts are not allowed. The only exception are emergency break-glass accounts, whose "
                "credentials are stored in the company password vault and whose use is logged and reviewed.",
            ]),
            ("Passwords and Authentication", [
                "Passwords must have at least 14 characters. Passwords are not changed on a fixed schedule; a change is "
                "required when there is any sign that a password may be compromised. Company passwords must not be "
                "reused for private accounts. Employees should use the company password manager.",
                "Multi-factor authentication (MFA) is mandatory for remote access through VPN, for email and "
                "collaboration tools used from outside the company network, for all administrator accounts and for all "
                "cloud services that hold company data.",
            ]),
            ("Endpoint Security", [
                "All laptops use full-disk encryption. Screens lock automatically after 10 minutes of inactivity. "
                "The endpoint detection and response (EDR) agent must not be disabled or removed.",
                "USB mass storage devices are blocked by default. An exception can be approved by the CISO for a "
                "specific business need for a maximum of 90 days. Personal devices may access company email only "
                "through the managed mobile app.",
            ]),
            ("Patch and Vulnerability Management", [
                "Security updates are installed within the following deadlines after release: critical vulnerabilities "
                "(CVSS 9.0 and higher) within 14 days, high vulnerabilities (CVSS 7.0 to 8.9) within 30 days, medium "
                "vulnerabilities within 90 days and low vulnerabilities with the next regular release.",
                "If a vulnerability is known to be actively exploited, for example because it is listed in the CISA "
                "Known Exploited Vulnerabilities catalog, the update must be installed within 72 hours.",
                "Servers are patched in the monthly maintenance window on the second Saturday of each month. Systems "
                "that cannot be patched, such as production machines in the OT network, require documented compensating "
                "controls and an exception approved by the CISO. Such exceptions are reviewed every 6 months.",
            ]),
            ("Network Security", [
                "The production (OT) network is separated from the office network. Remote access by machine vendors to "
                "the OT network is only allowed through the central jump host, and these sessions are recorded. "
                "The guest Wi-Fi is isolated from all internal networks.",
            ]),
            ("Backup and Recovery", [
                "Critical systems are backed up daily and backups are kept for 30 days. At least one backup copy is "
                "stored offline or in immutable storage. Restoring from backup is tested every quarter.",
            ]),
            ("Exceptions and Enforcement", [
                "Exceptions to this policy are requested with the exception request form and approved by the CISO for a "
                "maximum of 12 months. Violations of this policy may lead to disciplinary action.",
            ]),
        ],
    },
    "IT-04_Acceptable_Use_and_Generative_AI_Policy": {
        "title": "Acceptable Use and Generative AI Policy",
        "meta": "Document IT-04, version 2.0, owners: Chief Information Officer and Data Protection Officer, effective 1 January 2026",
        "sections": [
            ("Purpose and Scope", [
                "This policy describes how company IT resources, including generative AI tools, may be used. It applies "
                "to all employees and contractors who use company devices, accounts or data.",
            ]),
            ("General Acceptable Use", [
                "Company IT resources are provided for business purposes. Limited personal use is allowed if it does not "
                "interfere with work, security or other employees. Illegal content, cryptocurrency mining and the "
                "installation of unlicensed software are prohibited. Software is installed only through the company "
                "software center.",
            ]),
            ("Email and Collaboration", [
                "Confidential documents are shared with external parties only through the company file sharing service, "
                "using links that expire after at most 30 days. Automatic forwarding of company email to private email "
                "accounts is prohibited.",
            ]),
            ("Approved Generative AI Tools", [
                "Only the generative AI tools on the approved AI tools list on the intranet may be used with company "
                "information. At the moment the list contains the enterprise chat assistant, which is covered by a data "
                "processing agreement, and the code assistant for software developers.",
                "Public, free AI tools may only be used with information that is already public.",
            ]),
            ("Data Classification and AI Tools", [
                "Company information is classified as Public, Internal, Confidential or Strictly Confidential. Internal "
                "information may be entered only into approved AI tools. Confidential information may be entered only "
                "into approved AI tools that process data in the EU.",
                "Strictly Confidential information, such as unreleased product designs, merger and acquisition plans or "
                "salary data, must never be entered into any generative AI tool. Personal data of customers, applicants "
                "or employees must never be entered into a generative AI tool unless a data protection impact assessment "
                "(DPIA) has approved that specific use case.",
            ]),
            ("Human Review and Labelling", [
                "AI output must be reviewed by a person before it is used externally or as the basis for a decision. "
                "Images and texts generated with AI that are published externally must be labelled as AI-generated. "
                "Source code produced with an AI assistant goes through the normal code review and the open source "
                "license scan.",
            ]),
            ("New AI Use Cases", [
                "Every new AI use case must be registered with the AI Governance Board before development starts. The "
                "board classifies the use case under the EU AI Act. High-risk use cases, for example in recruiting or in "
                "the evaluation of employee performance, need the approval of the board and a DPIA.",
                "Prohibited practices, such as recognising the emotions of employees, will not be approved. The board "
                "meets monthly and decides on complete requests within 20 working days.",
            ]),
            ("Monitoring", [
                "The company may monitor the use of IT systems for security purposes in line with the agreement with "
                "the works council. IT logs are not used to monitor the performance of individual employees.",
            ]),
            ("Violations", [
                "Violations of this policy may lead to the withdrawal of access rights and to disciplinary action.",
            ]),
        ],
    },
    "DP-02_Data_Protection_and_Retention_Policy": {
        "title": "Data Protection and Retention Policy",
        "meta": "Document DP-02, version 1.4, owner: Data Protection Officer (DPO), effective 15 September 2025",
        "sections": [
            ("Purpose and Scope", [
                "This policy implements the requirements of the EU General Data Protection Regulation (GDPR) and the "
                "German Federal Data Protection Act (BDSG) for all processing of personal data by the company.",
            ]),
            ("Principles", [
                "Personal data is processed lawfully, for specified purposes, limited to what is necessary, kept accurate, "
                "stored no longer than needed and protected by appropriate security measures.",
            ]),
            ("Records of Processing Activities", [
                "Each department maintains its entries in the record of processing activities (Art. 30 GDPR) and updates "
                "them within 30 days of any change to a processing activity.",
            ]),
            ("Data Protection Impact Assessment", [
                "A data protection impact assessment (DPIA) is carried out before any new processing that is likely to "
                "result in a high risk to the rights and freedoms of people. The DPO must be consulted. A DPIA is "
                "reviewed every 3 years or when the processing changes significantly.",
            ]),
            ("Retention Periods", [
                "Personal data is deleted when the retention period ends. The standard retention periods are:",
                "• Data of unsuccessful applicants: deleted 6 months after the end of the application process, unless "
                "the applicant agreed to be kept in the talent pool, which is limited to 2 years.",
                "• Invoices and accounting vouchers: 8 years. Ledgers and annual financial statements: 10 years.",
                "• CCTV recordings at the plant gates: 72 hours, unless they are needed to investigate an incident.",
                "• Visitor registration logs: 3 months.",
                "• System logs that contain personal data: 90 days. Security logs: 12 months.",
                "• Customer contact data in the CRM: 3 years after the last business contact.",
            ]),
            ("Deletion and Disposal", [
                "Deletion is automated where possible and is logged. Paper documents with personal data are shredded "
                "to at least security level P-4 according to DIN 66399. Storage media are wiped or destroyed by the "
                "certified disposal provider.",
            ]),
            ("Data Subject Requests", [
                "Requests from people exercising their rights, such as access, correction or deletion, are forwarded to "
                "the DPO within 2 working days. The company answers within one month of receipt, as required by "
                "Art. 12(3) GDPR.",
            ]),
            ("Personal Data Breaches", [
                "Every suspected personal data breach must be reported to the DPO and to IT Security immediately and in "
                "any case within 24 hours of discovery. The DPO decides whether the breach must be notified to the "
                "supervisory authority, which has to happen within 72 hours (Art. 33 GDPR). Affected people are "
                "informed when the breach is likely to result in a high risk to them (Art. 34 GDPR).",
            ]),
            ("International Transfers", [
                "Personal data is transferred outside the EU and EEA only on the basis of an adequacy decision or the EU "
                "standard contractual clauses, and only with the approval of the DPO.",
            ]),
        ],
    },
    "IS-07_Incident_Response_Procedure": {
        "title": "Incident Response Procedure",
        "meta": "Document IS-07, version 2.1, owner: Chief Information Security Officer (CISO), effective 1 June 2026",
        "sections": [
            ("Purpose", [
                "This procedure describes how security incidents are reported, classified and handled, so that damage "
                "is limited and operations are restored quickly.",
            ]),
            ("What to Report", [
                "Employees report suspected phishing, malware, unauthorised access, unusual system behaviour and lost or "
                "stolen devices to the security hotline (extension 4444) or the security mailbox. A lost or stolen laptop "
                "or phone must be reported within 1 hour of noticing the loss.",
            ]),
            ("Severity Levels and Response Times", [
                "P1 Critical: production is stopped, ransomware is active, or a breach of sensitive data is confirmed. "
                "P1 incidents are acknowledged within 15 minutes, around the clock. An incident manager is assigned and "
                "the CISO and management are informed within 1 hour.",
                "P2 High: a single critical system is affected or a data breach is suspected. Response within 1 hour "
                "during business hours.",
                "P3 Medium: limited impact, for example malware on a single computer that has been contained. Response "
                "within 4 business hours.",
                "P4 Low: policy violations or phishing emails that nobody clicked. Response on the next business day.",
            ]),
            ("First Steps for Employees", [
                "If you suspect ransomware on your computer, disconnect it from the network by unplugging the cable and "
                "switching off Wi-Fi. Do not switch the computer off, because this can destroy evidence, and do not try "
                "to fix the problem yourself. Call the security hotline.",
                "Report suspicious emails with the Report Phishing button and do not click any links. If you already "
                "clicked a link or entered your password, report it immediately and change your password.",
            ]),
            ("Containment and Eradication", [
                "The security team isolates affected computers through the EDR system, preserves evidence (where possible "
                "a memory image before any restart), resets compromised credentials and removes the cause of the incident "
                "before systems are restored from clean backups.",
            ]),
            ("Communication and Reporting Obligations", [
                "Only the Communications department speaks to the media. Customers are informed by account management "
                "after approval by Legal.",
                "Where the company is subject to the NIS2 directive, significant incidents are reported to the BSI: an "
                "early warning within 24 hours of becoming aware of the incident, an incident notification within "
                "72 hours, and a final report within one month. Personal data breaches follow the Data Protection and "
                "Retention Policy (DP-02).",
            ]),
            ("Post-Incident Review", [
                "A post-incident review is held within 5 working days after a P1 or P2 incident is closed. Lessons "
                "learned and follow-up actions are tracked in the risk register.",
            ]),
            ("Exercises", [
                "The incident response team runs a tabletop exercise twice a year. At least one exercise per year uses "
                "a ransomware scenario.",
            ]),
        ],
    },
}


def build_pdf(name, policy):
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=7)
    heading = ParagraphStyle("heading", parent=styles["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=6)
    meta = ParagraphStyle("meta", parent=body, fontSize=9, textColor="#52514e")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(2 * cm, 1.2 * cm, f"{COMPANY} - {policy['title']} - Internal")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    story = [Paragraph(policy["title"], styles["Title"]), Paragraph(f"{COMPANY}. {policy['meta']}.", meta),
             Spacer(1, 8)]
    for number, (title, paragraphs) in enumerate(policy["sections"], start=1):
        story.append(Paragraph(f"{number}. {title}", heading))
        story += [Paragraph(p, body) for p in paragraphs]

    path = OUT_DIR / f"{name}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, title=policy["title"], author=COMPANY,
                            leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, policy in POLICIES.items():
        print(build_pdf(name, policy))
