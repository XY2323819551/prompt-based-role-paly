

### Solution

How do I delete my payment method?

Updated over a week ago
We keep your payment method on file to cover any outstanding charges on your account. To stop charges to your payment method, please follow the steps below.

**ChatGPT**

You can cancel your ChatGPT Plus subscription to stop further charges at any time:
Click on 'My Plan' in the ChatGPT sidebar.
Click on 'Manage my subscription' in the pop-up window.
Select 'Cancel Plan'.
Please note that your cancellation will take effect the day after the next billing date, and you can continue using our services until then. To avoid being charged for your next billing period, please cancel your subscription at least 24 hours before your next billing date.

**API**

We'll need to keep a payment method on file to account for any outstanding usage costs. You're welcome to cancel your pay-as-you-go service, by clicking 'Cancel paid account' in your billing overview. After the current month's invoice has been issued, the current card will no longer be charged.
If you'd like to continue using the service, add a new payment method in the billing overview page and select 'Set as default'. You'll then be able to delete the old payment method.





### Routine

1. Verify the customer's account.
  1. Politely ask the customer for their email address or account ID to locate their account.

  2.  `call the verify_customer_account(email_or_account_id)`.

2. Verify the customer's identity.
  1. Politely ask the customer to provide security information to confirm their identity (e.g., the last four digits of the payment method on file).
  2.  `call the verify_customer_identity(account_id, security_information)`.
  3. If the customer's identity cannot be verified, then:
     1. Inform the customer that we are unable to proceed without identity verification for security reasons.
     2. Provide guidance on how they can verify their identity.
     3. Proceed to step 6.

3. Determine the customer's account type.
    a. `call the check_account_type(account_id)`.

4. If the customer is a ChatGPT Plus subscriber, then:
    a. Ask the customer if they would like assistance with canceling their ChatGPT Plus subscription.
    b. If the customer agrees, then:

  - `call the cancel_subscription(account_id)`.
  - Inform the customer that their subscription has been canceled and the cancellation will take effect the day after the next billing date.
  - Remind the customer that they can continue using our services until then.

  c. Else:

  - Provide the following steps for the customer to cancel their subscription:
  - Click on **'My Plan'** in the ChatGPT sidebar.
  - Click on **'Manage my subscription'** in the pop-up window.
  - Select **'Cancel Plan'**.
  - Inform the customer about the cancellation effective date and continued access until then.
  - Advise the customer to cancel at least 24 hours before their next billing date to avoid being charged for the next billing period.
    
5. Else if the customer is an API user, then:
    a. Inform the customer that we need to keep a payment method on file to account for any outstanding usage costs.
    b. Ask the customer if they would like assistance with canceling their pay-as-you-go service.
    c. If the customer agrees, then:

  - `call the cancel_paid_account(account_id)`.
  - Inform the customer that after the current month's invoice has been issued, the current card will no longer be charged.

  d. Else:

  - Provide the following steps for the customer to cancel their pay-as-you-go service:
  - Go to the **billing overview** page.
  - Click on **'Cancel paid account'**.
  - Inform the customer that after the current month's invoice has been issued, the current card will no longer be charged.

  e. If the customer wants to continue using the service but change the payment method:

  - Ask the customer if they would like assistance with adding a new payment method and setting it as default.
  - If the customer agrees:

    - Politely request the new payment method details.
    - `call the add_payment_method(account_id, payment_details)`.
    - `call the set_default_payment_method(account_id, new_payment_method_id)`.
    - `call the delete_payment_method(account_id, old_payment_method_id)`.
    - Inform the customer that the old payment method has been deleted and the new one is set as default.
  - Else:

    - Instruct the customer to add a new payment method in the billing overview page.

    - Ask them to select **'Set as default'** for the new payment method.

    - Inform them that they can then delete the old payment method.

6. Ask the customer if there is anything else you can assist them with.
  
7. `call the case_resolution()`.


---

**Function Definitions:**

- `verify_customer_account(email_or_account_id)`: Verifies the customer's account using their email address or account ID.
  **Parameters:** `email_or_account_id`
  
- `verify_customer_identity(account_id, security_information)`: Confirms the customer's identity using security information.
  **Parameters:** `account_id`, `security_information`
  
- `check_account_type(account_id)`: Retrieves the customer's account type (ChatGPT Plus subscriber or API user).
  **Parameters:** `account_id`
  
- `cancel_subscription(account_id)`: Cancels the ChatGPT Plus subscription for the customer.
  **Parameters:** `account_id`
  
- `cancel_paid_account(account_id)`: Cancels the API pay-as-you-go service for the customer.
  **Parameters:** `account_id`
  
- `add_payment_method(account_id, payment_details)`: Adds a new payment method to the customer's account.
  **Parameters:** `account_id`, `payment_details`
  
- `set_default_payment_method(account_id, payment_method_id)`: Sets a payment method as the default for the customer.
  **Parameters:** `account_id`, `payment_method_id`
  
- `delete_payment_method(account_id, payment_method_id)`: Deletes a payment method from the customer's account.
  **Parameters:** `account_id`, `payment_method_id`
  
- `case_resolution()`: Resolves the case and marks it as completed.