/**
 *Submitted for verification at Etherscan.io on 2019-02-08
*/

pragma solidity ^0.4.24;

/**
 * @title SafeMath
 * @dev Math operations with safety checks that throw on error
 */
library SafeMath {
  function mul(uint256 a, uint256 b) internal pure returns (uint256) {
    if (a == 0) {
      return 0;
    }

    uint256 c = a * b;
    require(c / a == b);

    return c;
  }

  function div(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b > 0);
    uint256 c = a / b;
    return c;
  }

  function sub(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b <= a);
    uint256 c = a - b;
    return c;
  }

  function add(uint256 a, uint256 b) internal pure returns (uint256) {
    uint256 c = a + b;
    require(c >= a);
    return c;
  }

  /**
  * @dev Divides two unsigned integers and returns the remainder (unsigned integer modulo),
  * reverts when dividing by zero.
  */
  function mod(uint256 a, uint256 b) internal pure returns (uint256) {
    require(b != 0);
    return a % b;
  }
}

/**
 * @title Ownable
 * @dev The Ownable contract has an owner address, and provides basic authorization control
 * functions, this simplifies the implementation of "user permissions".
 */
contract Ownable {
  address public owner;

  /**
    * @dev The Ownable constructor sets the original `owner` of the contract to the sender
    * account.
    */
  constructor() public {
    owner = msg.sender;
  }

  /**
    * @dev Throws if called by any account other than the owner.
    */
  modifier onlyOwner() {
    require(
      msg.sender == owner,
      "msg.sender is not owner"
    );
    _;
  }

  /**
  * @dev Allows the current owner to transfer control of the contract to a newOwner.
  * @param newOwner The address to transfer ownership to.
  */
  function transferOwnership(address newOwner)
    public
    onlyOwner
    returns (bool)
  {
    if (newOwner != address(0) && newOwner != owner) {
      owner = newOwner;
      return true;
    } else {
      return false;
    }
  }
}

/**
 * @title ERC20Basic
 * @dev Simpler version of ERC20 interface
 * @dev see https://github.com/ethereum/EIPs/issues/20
 */
contract ERC20Basic {
  uint public _totalSupply;
  function totalSupply() public view returns (uint);
  function balanceOf(address who) public view returns (uint);
  function transfer(address to, uint value) public returns (bool);
  event Transfer(address indexed from, address indexed to, uint value);
}

/**
 * @title ERC20 interface
 * @dev see https://github.com/ethereum/EIPs/issues/20
 */
contract ERC20 is ERC20Basic {
  function allowance(
    address owner,
    address spender) public view returns (uint);
  function transferFrom(
    address from,
    address to,
    uint value
  )
    public returns (bool);
  function approve(address spender, uint value) public returns (bool);
  event Approval(address indexed owner, address indexed spender, uint value);
}

/**
 * @title WhiteList
 * @dev All the addresses whitelisted will not pay the fee for transfer and transferFrom.
 */

contract WhiteList is Ownable {
  mapping(address => bool) public whitelist;

  function addToWhitelist (address _address) public onlyOwner returns (bool) {
    whitelist[_address] = true;
    return true;
  }

  function removeFromWhitelist (address _address)
    public onlyOwner returns (bool) 
  {
    whitelist[_address] = false;
    return true;
  }
}

/**
 * @title Basic token
 * @dev Basic version of StandardToken, with no allowances.
 */
contract BasicToken is WhiteList, ERC20Basic {
  using SafeMath for uint;

  mapping(address => uint) public balances;

  /**
  * @dev additional variables for use if transaction fees ever became necessary
  */
  uint public basisPointsRate = 0;
  uint public maximumFee = 0;

  /**
  * @dev Fix for the ERC20 short address attack.
  */
  modifier onlyPayloadSize(uint size) {
    require(
      !(msg.data.length < size + 4),
      "msg.data length is wrong"
    );
    _;
  }

  /**
  * @dev transfer token for a specified address
  * @param _to The address to transfer to.
  * @param _value The amount to be transferred.
  */
  function transfer(address _to, uint _value)
    public
    onlyPayloadSize(2 * 32)
    returns (bool)
  {
    uint fee = whitelist[msg.sender]
      ? 0
      : (_value.mul(basisPointsRate)).div(10000);

    if (fee > maximumFee) {
      fee = maximumFee;
    }
    uint sendAmount = _value.sub(fee);
    balances[msg.sender] = balances[msg.sender].sub(_value);
    balances[_to] = balances[_to].add(sendAmount);
    if (fee > 0) {
      balances[owner] = balances[owner].add(fee);
      emit Transfer(msg.sender, owner, fee);
      return true;
    }
    emit Transfer(msg.sender, _to, sendAmount);
    return true;
  }

    /**
    * @dev Gets the balance of the specified address.
    * @param _owner The address to query the the balance of.
    * @return An uint representing the amount owned by the passed address.
    */
  function balanceOf(address _owner) public view returns (uint balance) {
    return balances[_owner];
  }

}

/**
 * @title Standard ERC20 token
 *
 * @dev Implementation of the basic standard token.
 * @dev https://github.com/ethereum/EIPs/issues/20
 * @dev Based oncode by FirstBlood: https://github.com/Firstbloodio/token/blob/master/smart_contract/FirstBloodToken.sol
 */
contract StandardToken is BasicToken, ERC20 { 

  mapping (address => mapping (address => uint)) public allowed;

  uint public constant MAX_UINT = 2**256 - 1;

  /**
  * @dev Transfer tokens from one address to another
  * @param _from address The address which you want to send tokens from
  * @param _to address The address which you want to transfer to
  * @param _value uint the amount of tokens to be transferred
  */
  function transferFrom(
    address _from,
    address _to,
    uint
    _value
  )
    public
    onlyPayloadSize(3 * 32)
    returns (bool)
  {
    uint _allowance = allowed[_from][msg.sender];

    // Check is not needed because sub(_allowance, _value) will already throw if this condition is not met
    // if (_value > _allowance) throw;

    uint fee = whitelist[msg.sender]
      ? 0
      : (_value.mul(basisPointsRate)).div(10000);
    if (fee > maximumFee) {
      fee = maximumFee;
    }
    if (_allowance < MAX_UINT) {
      allowed[_from][msg.sender] = _allowance.sub(_value);
    }
    uint sendAmount = _value.sub(fee);
    balances[_from] = balances[_from].sub(_value);
    balances[_to] = balances[_to].add(sendAmount);
    if (fee > 0) {
      balances[owner] = balances[owner].add(fee);
      emit Transfer(_from, owner, fee);
      return true;
    }
    emit Transfer(_from, _to, sendAmount);
    return true;
  }

  /**
  * @dev Approve the passed address to spend the specified amount of tokens on behalf of msg.sender.
  * @param _spender The address which will spend the funds.
  * @param _value The amount of tokens to be spent.
  */
  function approve(
    address _spender,
    uint _value
  )
    public
    onlyPayloadSize(2 * 32)
    returns (bool)
  {
    // To change the approve amount you first have to reduce the addresses`
    //  allowance to zero by calling `approve(_spender, 0)` if it is not
    //  already 0 to mitigate the race condition described here:
    //  https://github.com/ethereum/EIPs/issues/20#issuecomment-263524729
    require(
      !((_value != 0) && (allowed[msg.sender][_spender] != 0)),
      "Canont approve 0 as amount"
    );

    allowed[msg.sender][_spender] = _value;
    emit Approval(msg.sender, _spender, _value);
    return true;
  }

  /**
  * @dev Function to check the amount of tokens than an owner allowed to a spender.
  * @param _owner address The address which owns the funds.
  * @param _spender address The address which will spend the funds.
  * @return A uint specifying the amount of tokens still available for the spender.
  */
  function allowance(address _owner, address _spender)
    public
    view
    returns (uint remaining) 
  {
    return allowed[_owner][_spender];
  }
}

/**
 * @title Pausable
 *
 * @dev Base contract which allows children to implement an emergency stop mechanism.
 */
contract Pausable is Ownable {
  event Pause();
  event Unpause();

  bool public paused = false;


  /**
   * @dev Modifier to make a function callable only when the contract is not paused.
   */
  modifier whenNotPaused() {
    require(!paused, "paused is true");
    _;
  }

  /**
   * @dev Modifier to make a function callable only when the contract is paused.
   */
  modifier whenPaused() {
    require(paused, "paused is false");
    _;
  }

  /**
   * @dev Called by the owner to pause, triggers stopped state
   * @return Operation succeeded.
   */
  function pause()
    public
    onlyOwner
    whenNotPaused
    returns (bool) 
  {
    paused = true;
    emit Pause();
    return true;
  }

  /**
   * @dev Called by the owner to unpause, returns to normal state
   */
  function unpause()
    public
    onlyOwner
    whenPaused
    returns (bool)
  {
    paused = false;
    emit Unpause();
    return true;
  }
}

/**
 * @title BlackList
 *
 * @dev Base contract which allows the owner to blacklist a stakeholder and destroy its tokens.
 */
contract BlackList is Ownable, BasicToken {

  mapping (address => bool) public isBlackListed;

  event DestroyedBlackFunds(address _blackListedUser, uint _balance);
  event AddedBlackList(address _user);
  event RemovedBlackList(address _user);

  /**
   * @dev Add address to blacklist.
   * @param _evilUser Address to be blacklisted.
   * @return Operation succeeded.
   */
  function addBlackList (address _evilUser)
    public
    onlyOwner
    returns (bool)
  {
    isBlackListed[_evilUser] = true;
    emit AddedBlackList(_evilUser);
    return true;
  }

  /**
   * @dev Remove address from blacklist.
   * @param _clearedUser Address to removed from blacklist.
   * @return Operation succeeded.
   */
  function removeBlackList (address _clearedUser)
    public
    onlyOwner
    returns (bool)
  {
    isBlackListed[_clearedUser] = false;
    emit RemovedBlackList(_clearedUser);
    return true;
  }

  /**
   * @dev Destroy funds of the blacklisted user.
   * @param _blackListedUser Address of whom to destroy the funds.
   * @return Operation succeeded.
   */
  function destroyBlackFunds (address _blackListedUser)
    public
    onlyOwner
    returns (bool)
  {
    require(isBlackListed[_blackListedUser], "User is not blacklisted");
    uint dirtyFunds = balanceOf(_blackListedUser);
    balances[_blackListedUser] = 0;
    _totalSupply -= dirtyFunds;
    emit DestroyedBlackFunds(_blackListedUser, dirtyFunds);
    return true;
  }
}

/**
 * @title UpgradedStandardToken
 *
 * @dev Interface to submit calls from the current SC to a new one.
 */
contract UpgradedStandardToken is StandardToken{
  /**
   * @dev Methods called by the legacy contract
   * and they must ensure msg.sender to be the contract address.
   */
  function transferByLegacy(
    address from,
    address to,
    uint value) public returns (bool);
  function transferFromByLegacy(
    address sender,
    address from,
    address spender,
    uint value) public returns (bool);

  function approveByLegacy(
    address from,
    address spender,
    uint value) public returns (bool);
}

/**
 * @title BackedToken
 *
 * @dev ERC20 token backed by some asset periodically audited reserve.
 */
contract BackedToken is Pausable, StandardToken, BlackList {

  string public name;
  string public symbol;
  uint public decimals;
  address public upgradedAddress;
  bool public deprecated;

  // Called when new token are issued
  event Issue(uint amount);
  // Called when tokens are redeemed
  event Redeem(uint amount);
  // Called when contract is deprecated
  event Deprecate(address newAddress);
  // Called if contract ever adds fees
  event Params(uint feeBasisPoints, uint maxFee);

  /**
   * @dev Constructor.
   * @param _initialSupply Initial total supply.
   * @param _name Token name.
   * @param _symbol Token symbol.
   * @param _decimals Token decimals.
   */
  constructor (
    uint _initialSupply,
    string _name,
    string _symbol,
    uint _decimals
  ) public {
    _totalSupply = _initialSupply;
    name = _name;
    symbol = _symbol;
    decimals = _decimals;
    balances[owner] = _initialSupply;
    deprecated = false;
  }

  /**
   * @dev Revert whatever no named function is called.
   */
  function() public payable {
    revert("No specific function has been called");
  }

  /**
   * @dev ERC20 overwritten functions.
   */

  function transfer(address _to, uint _value)
    public whenNotPaused returns (bool) 
  {
    require(
      !isBlackListed[msg.sender],
      "Transaction recipient is blacklisted"
    );
    if (deprecated) {
      return UpgradedStandardToken(upgradedAddress).transferByLegacy(msg.sender, _to, _value);
    } else {
      return super.transfer(_to, _value);
    }
  }

  function transferFrom(
    address _from,
    address _to,
    uint _value
  )
    public
    whenNotPaused
    returns (bool)
  {
    require(!isBlackListed[_from], "Tokens owner is blacklisted");
    if (deprecated) {
      return UpgradedStandardToken(upgradedAddress).transferFromByLegacy(
        msg.sender,
        _from,
        _to,
        _value
      );
    } else {
      return super.transferFrom(_from, _to, _value);
    }
  }

  function balanceOf(address who) public view returns (uint) {
    if (deprecated) {
      return UpgradedStandardToken(upgradedAddress).balanceOf(who);
    } else {
      return super.balanceOf(who);
    }
  }

  function approve(
    address _spender,
    uint _value
  ) 
    public
    onlyPayloadSize(2 * 32)
    returns (bool)
  {
    if (deprecated) {
      return UpgradedStandardToken(upgradedAddress).approveByLegacy(msg.sender, _spender, _value);
    } else {
      return super.approve(_spender, _value);
    }
  }

  function allowance(
    address _owner,
    address _spender
  )
    public
    view
    returns (uint remaining) 
  {
    if (deprecated) {
      return StandardToken(upgradedAddress).allowance(_owner, _spender);
    } else {
      return super.allowance(_owner, _spender);
    }
  }

  function totalSupply() public view returns (uint) {
    if (deprecated) {
      return StandardToken(upgradedAddress).totalSupply();
    } else {
      return _totalSupply;
    }
  }

  /**
   * @dev Issue tokens. These tokens are added to the Owner address and to the _totalSupply.
   * @param amount Amount of the token to be issued to the owner balance adding it to the _totalSupply.
   * @return Operation succeeded.
   */
  function issue(uint amount)
    public
    onlyOwner
    returns (bool)
  {
    require(
      _totalSupply + amount > _totalSupply,
      "Wrong amount to be issued referring to _totalSupply"
    );

    require(
      balances[owner] + amount > balances[owner],
      "Wrong amount to be issued referring to owner balance"
    );

    balances[owner] += amount;
    _totalSupply += amount;
    emit Issue(amount);
    return true;
  }

  /**
   * @dev Redeem tokens. These tokens are withdrawn from the Owner address.
   * The balance must be enough to cover the redeem or the call will fail.
   * @param amount Amount of the token to be subtracted from the _totalSupply and the Owner balance.
   * @return Operation succeeded.
   */
  function redeem(uint amount)
    public
    onlyOwner
    returns (bool)
  {
    require(
      _totalSupply >= amount,
      "Wrong amount to be redeemed referring to _totalSupply"
    );
    require(
      balances[owner] >= amount,
      "Wrong amount to be redeemed referring to owner balance"
    );
    _totalSupply -= amount;
    balances[owner] -= amount;
    emit Redeem(amount);
    return true;
  }

  /**
   * @dev Set the current SC as deprecated.
   * @param _upgradedAddress The new SC address to be pointed from this SC.
   * @return Operation succeeded.
   */
  function deprecate(address _upgradedAddress)
    public
    onlyOwner
    returns (bool)
  {
    deprecated = true;
    upgradedAddress = _upgradedAddress;
    emit Deprecate(_upgradedAddress);
    return true;
  }

  /**
   * @dev Set fee params. The params has an hardcoded limit.
   * @param newBasisPoints The maker order object.
   * @param newMaxFee The amount of tokens going to the taker.
   * @return Operation succeeded.
   */
  function setParams(
    uint newBasisPoints,
    uint newMaxFee
  ) 
    public
    onlyOwner 
    returns (bool) 
  {
    // Ensure transparency by hardcoding limit beyond which fees can never be added
    require(
      newBasisPoints < 20,
      "newBasisPoints amount bigger than hardcoded limit"
    );
    require(
      newMaxFee < 50,
      "newMaxFee amount bigger than hardcoded limit"
    );
    basisPointsRate = newBasisPoints;
    maximumFee = newMaxFee.mul(10**decimals);
    emit Params(basisPointsRate, maximumFee);
    return true;
  }

  /**
   * @dev Selfdestruct the contract. Callable only from the owner.
   */
  function kill()
    public
    onlyOwner 
  {
    selfdestruct(owner);
  }
}