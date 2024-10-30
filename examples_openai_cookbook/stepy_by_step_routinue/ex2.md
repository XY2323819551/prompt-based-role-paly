

## Business Associate Agreement



### Solution

**How can I get a Business Associate Agreement (BAA) with OpenAI?**

Information about HIPAA compliance for healthcare companies

The Health Insurance Portability and Accountability Act (HIPAA) is a U.S. federal law that requires privacy and security protections for protected health information (PHI). Our API platform can be a great fit for any covered entity or business associate looking to process protected health information, and we’d be happy to assist you in fulfilling your HIPAA compliance. To use our API platform, you’ll first need a BAA with OpenAI.

**How do I get started?**

If you require a BAA before you can use our API, email us at baa@openai.com with details about your company and use case.

Our team will respond within 1-2 business days. We review each BAA request on a case-by-case basis and may need additional information. The process is usually completed within a few business days.

**Can I get a BAA for ChatGPT?**

If you're interested in exploring a BAA for ChatGPT Enterprise, please contact sales.

**What happens if I’m not approved?**

We are able to approve most customers that request BAAs, but occasionally a use case doesn’t pass our team's evaluation. In that case, we’ll give feedback and context as to why that is and give you the opportunity to update your intended use of our API and re-apply.

**Are all API services covered by the BAA?**

No, only endpoints that are eligible for zero retention are covered by the BAA. You can see a list of those endpoints.

**Is an enterprise agreement requirement to sign a BAA?**

No, an enterprise agreement is not required to sign a BAA.





### Routine

1. Thank the customer and ask for clarification:

​	a. "Thank you for reaching out! Could you please specify whether you require a Business Associate Agreement (BAA) for using our API or for ChatGPT Enterprise?"

2. If the customer requires a BAA for the API, then:
   - a. Inform the customer: "To obtain a BAA for our API, please email baa@openai.com with details about your company and use case."
   - b. Inform the customer: "Our team will respond within 1-2 business days."
   - c. Inform the customer: "We review each BAA request on a case-by-case basis and may need additional information."
   - d. Inform the customer: "The process is usually completed within a few business days."
   - e. Inform the customer: "Please note that only endpoints eligible for zero data retention are covered by the BAA."
   - i. Call the `provide_list_of_zero_retention_endpoints` function.
   - f. Inform the customer: "An enterprise agreement is not required to sign a BAA."

3. If the customer requires a BAA for ChatGPT Enterprise, then:
   - a. Inform the customer: "To explore a BAA for ChatGPT Enterprise, please contact our sales team."
   - i. Call the `provide_sales_contact_information` function.

4. If the customer is not approved, then:
   - Inform the customer: "We are able to approve most customers that request BAAs, but occasionally a use case doesn't pass our team's evaluation."
   - Inform the customer: "In that case, we'll provide feedback and context as to why and give you the opportunity to update your intended use of our API and re-apply."

5. Ask the customer if there is anything else you can assist with:
   - "Is there anything else I can assist you with today?"

6. Call the `case_resolution` function.



\---

**Function Definitions:**

\- `provide_list_of_zero_retention_endpoints`:

\- **Purpose**: Provides the customer with a list of API endpoints that are eligible for zero data retention under the BAA.

\- **Parameters**: None.

\- `provide_sales_contact_information`:

\- **Purpose**: Provides the customer with contact information to reach our sales team for ChatGPT Enterprise inquiries.

\- **Parameters**: None.

\- `case_resolution`:

\- **Purpose**: Finalizes the case and marks it as resolved.

\- **Parameters**: None.






