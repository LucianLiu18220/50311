/**
 *Submitted for verification at Etherscan.io on 2018-06-18
*/

pragma solidity ^0.4.9;


contract ERC223 {
    uint public totalSupply;
    function balanceOf(address who) public view returns (uint);
  
    function name() public view returns (string _name);
    function symbol() public view returns (string _symbol);
    function decimals() public view returns (uint8 _decimals);
    function totalSupply() public view returns (uint256 _supply);

    function transfer(address to, uint value) public returns (bool ok);
    function transfer(address to, uint value, bytes data) public returns (bool ok);
    function transfer(address to, uint value, bytes data, string custom_fallback) public returns (bool ok);
  
    event Transfer(address indexed from, address indexed to, uint value, bytes indexed data);
}


contract ContractReceiver {                
    function tokenFallback(address _from, uint _value, bytes _data) public;
}

 /**
 * ERC223 token by Dexaran
 *
 * https://github.com/Dexaran/ERC223-token-standard
 */
 
 
contract SafeMath {
    uint256 constant public MAX_UINT256 =
    0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;

    function safeAdd(uint256 x, uint256 y) pure internal returns (uint256 z) {
        if (x > MAX_UINT256 - y) revert();
        return x + y;
    }

    function safeSub(uint256 x, uint256 y) pure internal returns (uint256 z) {
        if (x < y) revert();
        return x - y;
    }

    function safeMul(uint256 x, uint256 y) pure internal returns (uint256 z) {
        if (y == 0) return 0;
        if (x > MAX_UINT256 / y) revert();
        return x * y;
    }
}
 
contract GwbToken is ERC223, SafeMath {

    mapping(address => uint) balances;
    
    string public name = "GoWeb";
    string public symbol = "GWB";
    uint8 public decimals = 8;
    uint256 public totalSupply = 7500000000000000;

    // ERC20 compatible event
    event Transfer(address indexed from, address indexed to, uint value);
    
    constructor () public {
		balances[tx.origin] = totalSupply;
	}

    // Function to access name of token .
    function name() public view returns (string _name) {
        return name;
    }
    // Function to access symbol of token .
    function symbol() public view returns (string _symbol) {
        return symbol;
    }
    // Function to access decimals of token .
    function decimals() public view returns (uint8 _decimals) {
        return decimals;
    }
    // Function to access total supply of tokens .
    function totalSupply() public view returns (uint256 _totalSupply) {
        return totalSupply;
    }

    
    // Function that is called when a user or another contract wants to transfer funds .
    function transfer(address _to, uint _value, bytes _data, string _custom_fallback) public returns (bool success) {        
        if(isContract(_to)) {
            if (balanceOf(msg.sender) < _value) revert();
            balances[msg.sender] = safeSub(balanceOf(msg.sender), _value);
            balances[_to] = safeAdd(balanceOf(_to), _value);
            assert(_to.call.value(0)(bytes4(keccak256(_custom_fallback)), msg.sender, _value, _data));
            emit Transfer(msg.sender, _to, _value, _data);
            return true;
        }
        else {
            return transferToAddress(_to, _value, false, _data);
        }
    }  
    
  
    // Function that is called when a user or another contract wants to transfer funds .
    function transfer(address _to, uint _value, bytes _data) public returns (bool success) {        
        if(isContract(_to)) {
            return transferToContract(_to, _value, false, _data);
        }
        else {
            return transferToAddress(_to, _value, false, _data);
        }
    }  
    
    // Standard function transfer similar to ERC20 transfer with no _data .
    // Added due to backwards compatibility reasons .
    function transfer(address _to, uint _value) public returns (bool success) {
        
      //standard function transfer similar to ERC20 transfer with no _data
      //added due to backwards compatibility reasons
        bytes memory empty;
        if(isContract(_to)) {
            return transferToContract(_to, _value, true, empty);
        }
        else {
            return transferToAddress(_to, _value, true, empty);
        }
    }  
  
    
  
    //function that is called when transaction target is an address
    function transferToAddress(address _to, uint _value, bool isErc20Transfer, bytes _data) private returns (bool success) {
        if (balanceOf(msg.sender) < _value) revert();
        balances[msg.sender] = safeSub(balanceOf(msg.sender), _value);
        balances[_to] = safeAdd(balanceOf(_to), _value);
        if (isErc20Transfer)
            emit Transfer(msg.sender, _to, _value);
        else
            emit Transfer(msg.sender, _to, _value, _data);
        return true;
    }
    
    //function that is called when transaction target is a contract
    function transferToContract(address _to, uint _value, bool isErc20Transfer, bytes _data) private returns (bool success) {
        if (balanceOf(msg.sender) < _value) revert();
        balances[msg.sender] = safeSub(balanceOf(msg.sender), _value);
        balances[_to] = safeAdd(balanceOf(_to), _value);
        ContractReceiver receiver = ContractReceiver(_to);
        receiver.tokenFallback(msg.sender, _value, _data);
        if (isErc20Transfer)
            emit Transfer(msg.sender, _to, _value);
        else
            emit Transfer(msg.sender, _to, _value, _data);
        return true;
    }  
  
    //assemble the given address bytecode. If bytecode exists then the _addr is a contract.
    function isContract(address _addr) private view returns (bool is_contract) {
        uint length;
        assembly {
              //retrieve the size of the code on target address, this needs assembly
              length := extcodesize(_addr)
        }
        return (length>0);
    }
  
    function balanceOf(address _owner) public view returns (uint balance) {
      return balances[_owner];
    }
}