/**
 *Submitted for verification at Etherscan.io on 2016-04-28
*/

contract tokenRecipient { function receiveApproval(address _from, uint256 _value, address _token, bytes _extraData); }

    contract Nexium { 
        /* Public variables of the token */
        string public name;
        string public symbol;
        uint8 public decimals;

        /* This creates an array with all balances */
        mapping (address => uint256) public balanceOf;
        mapping (address => mapping (address => uint)) public allowance;
        mapping (address => mapping (address => uint)) public spentAllowance;

        /* This generates a public event on the blockchain that will notify clients */
        event Transfer(address indexed from, address indexed to, uint256 value);

        /* Initializes contract with initial supply tokens to the creator of the contract */
        function Nexium() {
            balanceOf[msg.sender] = 100000000000;              // Give the creator all initial tokens                    
            name = 'Nexium';                                   // Set the name for display purposes     
            symbol = 'NxC';                               // Set the symbol for display purposes    
            decimals = 3;                            // Amount of decimals for display purposes        
        }

        /* Send coins */
        function transfer(address _to, uint256 _value) {
            if (balanceOf[msg.sender] < _value) throw;           // Check if the sender has enough   
            if (balanceOf[_to] + _value < balanceOf[_to]) throw; // Check for overflows
            balanceOf[msg.sender] -= _value;                     // Subtract from the sender
            balanceOf[_to] += _value;                            // Add the same to the recipient            
            Transfer(msg.sender, _to, _value);                   // Notify anyone listening that this transfer took place
        }

        /* Allow another contract to spend some tokens in your behalf */

        function approveAndCall(address _spender, uint256 _value, bytes _extraData) returns (bool success) {
            allowance[msg.sender][_spender] = _value;     
            tokenRecipient spender = tokenRecipient(_spender);
            spender.receiveApproval(msg.sender, _value, this, _extraData);
			
			return true;
        }

        /* A contract attempts to get the coins */

        function transferFrom(address _from, address _to, uint256 _value) returns (bool success) {
            if (balanceOf[_from] < _value) throw;                 // Check if the sender has enough   
            if (balanceOf[_to] + _value < balanceOf[_to]) throw;  // Check for overflows
            if (spentAllowance[_from][msg.sender] + _value > allowance[_from][msg.sender]) throw;   // Check allowance
            balanceOf[_from] -= _value;                          // Subtract from the sender
            balanceOf[_to] += _value;                            // Add the same to the recipient            
            spentAllowance[_from][msg.sender] += _value;
            Transfer(msg.sender, _to, _value); 
			
			return true;
        } 

        /* This unnamed function is called whenever someone tries to send ether to it */
        function () {
            throw;     // Prevents accidental sending of ether
        }        
    }